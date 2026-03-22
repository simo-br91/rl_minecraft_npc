package com.aberrada.rlnpc;

import net.minecraft.core.BlockPos;
import net.minecraft.network.chat.Component;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.CropBlock;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.levelgen.Heightmap;

import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

public class EnvironmentManager {

    // ------------------------------------------------------------------
    // Constants
    // ------------------------------------------------------------------

    private static final double NAV_SUCCESS_DISTANCE  = 1.5;
    private static final double FARM_SUCCESS_DISTANCE = 1.05;
    private static final double STUCK_EPS             = 0.01;
    private static final int    STUCK_LIMIT           = 8;
    private static final double MAX_EPISODE_DISTANCE  = 30.0;

    private static final int FARM_X     = 6;
    private static final int FARM_Z     = 2;

    // ------------------------------------------------------------------
    // Fields
    // ------------------------------------------------------------------

    private final MinecraftServer server;
    private final EpisodeState    state;
    private final Random          rng = new Random();

    public EnvironmentManager(MinecraftServer server) {
        this.server = server;
        this.state  = new EpisodeState();
    }

    // ------------------------------------------------------------------
    // reset()
    // ------------------------------------------------------------------

    /**
     * Resets the environment for a new episode.
     * All world/entity manipulation is scheduled on the main server thread
     * via CompletableFuture to avoid race conditions with the game tick.
     */
    public synchronized String reset(String taskName) {
        ServerPlayer observer = getObserverPlayer();
        if (observer == null) return errorJson("No player found. Open a singleplayer world first.");

        CompletableFuture<String> future = new CompletableFuture<>();

        server.execute(() -> {
            try {
                ServerLevel level = observer.serverLevel();

                // Clear any blocks from the previous episode
                if ("farming".equals(state.taskName) && state.farmingSoilPos != null) {
                    clearFarmArea(level, state.farmingSoilPos.getX(), state.farmingSoilPos.getZ());
                }
                clearTaskArtifacts(level);

                state.setTask(taskName);

                double spawnX, spawnZ, spawnY;
                float  spawnYaw;
                RLNpcEntity agent = getOrCreateAgent(level);

                if ("farming".equals(state.taskName)) {
                    configureFarmingTask(level);
                    spawnX   = state.targetX + (rng.nextDouble() - 0.5) * 6.0;
                    spawnZ   = state.targetZ + (rng.nextDouble() - 0.5) * 6.0;
                    spawnY   = resolveStandY(level, spawnX, spawnZ);
                    spawnYaw = (float) (rng.nextDouble() * 360.0 - 180.0);
                } else {
                    configureNavigationTask(level);
                    spawnX   = (rng.nextDouble() - 0.5) * 2.0;
                    spawnZ   = (rng.nextDouble() - 0.5) * 2.0;
                    spawnY   = resolveStandY(level, spawnX, spawnZ);
                    spawnYaw = (float) (rng.nextDouble() * 360.0 - 180.0);
                    // Place 1-block wall obstacles on the direct path
                    placeNavigationObstacles(level, spawnX, spawnZ);
                }

                placeAgent(agent, spawnX, spawnY, spawnZ, spawnYaw);
                placeMarker(level);

                double initDist = distanceToTarget(agent);
                state.reset(initDist);

                double[]          obs  = ObservationBuilder.build(agent, state);
                Map<String,Object> info = buildInfo(false, initDist, taskProgress(agent));
                future.complete(jsonResponse(obs, 0.0, false, false, info));

            } catch (Exception e) {
                RLNpcMod.LOGGER.error("reset() error", e);
                future.completeExceptionally(e);
            }
        });

        try {
            return future.get(10, TimeUnit.SECONDS);
        } catch (Exception e) {
            return errorJson("Reset timed out: " + e.getMessage());
        }
    }

    // ------------------------------------------------------------------
    // step()
    // ------------------------------------------------------------------

    /**
     * Executes one environment step on the main server thread.
     */
    public synchronized String step(int action) {
        ServerPlayer observer = getObserverPlayer();
        if (observer == null) return errorJson("No player found. Open a singleplayer world first.");

        CompletableFuture<String> future = new CompletableFuture<>();

        server.execute(() -> {
            try {
                ServerLevel level = observer.serverLevel();
                RLNpcEntity agent = getOrCreateAgent(level);

                // If the episode is already over, just return the terminal obs
                if (state.done) {
                    double cd = distanceToTarget(agent);
                    future.complete(jsonResponse(
                            ObservationBuilder.build(agent, state),
                            0.0, true, false,
                            buildInfo(state.success, cd, taskProgress(agent))));
                    return;
                }

                double beforeDistance = distanceToTarget(agent);
                state.lastInteractValid  = false;
                state.lastJumpedObstacle = false;

                boolean validAction;
                if (action == 3) {
                    validAction = handleInteract(level, agent);
                } else {
                    validAction = ActionExecutor.applyMovementAction(agent, action, state);
                }

                double currentDistance = distanceToTarget(agent);
                double delta           = beforeDistance - currentDistance;
                state.episodeStep++;

                if (Math.abs(delta) < STUCK_EPS) state.stuckSteps++;
                else                              state.stuckSteps = 0;

                boolean success   = computeSuccess(level, agent);
                boolean truncated = state.episodeStep >= state.maxSteps
                                 || distanceToTarget(agent) > MAX_EPISODE_DISTANCE;

                state.success = success;
                state.done    = success || truncated;

                if (!validAction) state.invalidActionCount++;

                double reward = computeReward(
                        agent, action, beforeDistance, currentDistance, validAction, success);
                state.prevDistance = currentDistance;

                double[]          obs  = ObservationBuilder.build(agent, state);
                Map<String,Object> info = buildInfo(success, currentDistance, taskProgress(agent));
                future.complete(jsonResponse(obs, reward, state.done, truncated, info));

            } catch (Exception e) {
                RLNpcMod.LOGGER.error("step() error", e);
                future.completeExceptionally(e);
            }
        });

        try {
            return future.get(5, TimeUnit.SECONDS);
        } catch (Exception e) {
            return errorJson("Step timed out: " + e.getMessage());
        }
    }

    // ------------------------------------------------------------------
    // Task configuration
    // ------------------------------------------------------------------

    private void configureNavigationTask(ServerLevel level) {
        double angle  = rng.nextDouble() * 2.0 * Math.PI;
        double dist   = 7.0 + rng.nextDouble() * 7.0;   // 7–14 blocks
        state.targetX = Math.round(Math.cos(angle) * dist);
        state.targetZ = Math.round(Math.sin(angle) * dist);
        state.targetY = resolveGroundY(level, state.targetX, state.targetZ) + 1.0;
    }

    /**
     * Places 1–2 stone blocks (1 block high) along the direct line from the
     * agent spawn (near origin) to the navigation target.  The obstacles are
     * placed at ~35% and (optionally) ~65% of the way along that line so that
     * the agent must learn to jump over them.
     */
    private void placeNavigationObstacles(ServerLevel level, double spawnX, double spawnZ) {
        int numObstacles = 1 + rng.nextInt(2); // 1 or 2

        double dx   = state.targetX - spawnX;
        double dz   = state.targetZ - spawnZ;
        double dist = Math.sqrt(dx * dx + dz * dz);

        if (dist < 4.0) return; // too close to bother

        double[] fractions = {0.35, 0.65};
        for (int i = 0; i < numObstacles; i++) {
            double frac = fractions[i] + (rng.nextDouble() - 0.5) * 0.08;
            int ox      = (int) Math.round(spawnX + dx * frac);
            int oz      = (int) Math.round(spawnZ + dz * frac);
            int groundY = getReferenceGroundY(level, ox, oz);

            BlockPos wallPos = new BlockPos(ox, groundY + 1, oz);
            // Avoid placing on top of the target marker or the agent spawn
            double wallDist = Math.sqrt(Math.pow(ox - state.targetX, 2) + Math.pow(oz - state.targetZ, 2));
            if (wallDist < 1.5) continue;

            level.setBlockAndUpdate(wallPos, Blocks.STONE.defaultBlockState());
            state.obstaclePositions.add(wallPos);
        }
    }

    private void configureFarmingTask(ServerLevel level) {
        int farmX   = FARM_X + rng.nextInt(7) - 3;
        int farmZ   = FARM_Z + rng.nextInt(7) - 3;
        int groundY = getReferenceGroundY(level, farmX, farmZ);

        BlockPos soilPos = new BlockPos(farmX, groundY, farmZ);
        BlockPos cropPos = soilPos.above();

        level.setBlockAndUpdate(soilPos, Blocks.FARMLAND.defaultBlockState());
        CropBlock wheat = (CropBlock) Blocks.WHEAT;
        level.setBlockAndUpdate(cropPos, wheat.getStateForAge(wheat.getMaxAge()));

        state.targetX        = farmX - 0.5;
        state.targetY        = cropPos.getY();
        state.targetZ        = farmZ + 0.5;
        state.farmingSoilPos = soilPos;
        state.farmingCropPos = cropPos;
    }

    // ------------------------------------------------------------------
    // Interact (farming harvest)
    // ------------------------------------------------------------------

    private boolean handleInteract(ServerLevel level, RLNpcEntity agent) {
        if (!"farming".equals(state.taskName)) return false;

        BlockPos   frontPos = ObservationBuilder.frontCropPos(agent);
        BlockState block    = level.getBlockState(frontPos);

        if (block.getBlock() instanceof CropBlock cropBlock && cropBlock.isMaxAge(block)) {
            server.execute(() -> level.destroyBlock(frontPos, true, agent));
            state.lastInteractValid = true;
            return true;
        }
        return false;
    }

    // ------------------------------------------------------------------
    // Success / reward
    // ------------------------------------------------------------------

    private boolean computeSuccess(ServerLevel level, RLNpcEntity agent) {
        if ("farming".equals(state.taskName)) {
            return state.lastInteractValid
                && state.farmingCropPos != null
                && distanceToTarget(agent) <= FARM_SUCCESS_DISTANCE
                && level.getBlockState(state.farmingCropPos).isAir();
        }
        return distanceToTarget(agent) <= NAV_SUCCESS_DISTANCE;
    }

    private double computeReward(
            RLNpcEntity agent, int action,
            double beforeDistance, double currentDistance,
            boolean validAction, boolean success) {

        double reward   = 0.0;
        double progress = beforeDistance - currentDistance;

        // Progress shaping: reward getting closer, penalise moving away
        reward += 0.15 * progress;

        // Small time penalty to encourage efficiency
        reward -= 0.015;

        if ("farming".equals(state.taskName)) {
            boolean cropInFront = ObservationBuilder.isMatureCropInFront(agent);
            boolean nearTarget  = currentDistance <= 1.10;

            if (nearTarget)   reward += 0.05;
            if (cropInFront)  reward += 0.10;   // reduced from 0.25 to avoid camping

            // Penalise drifting away once close
            if (beforeDistance <= 1.15 && currentDistance > beforeDistance + 1e-6)
                reward -= 0.10;

            // Interact rewards
            if (action == 3 && cropInFront)  reward += 2.0;
            if (action == 3 && !cropInFront) reward -= 0.05;
            if (state.lastInteractValid)      reward += 4.0;
        }

        // Jump bonus: encourage learning to use jump over obstacles
        if (state.lastJumpedObstacle) reward += 0.5;

        // Terminal bonus
        if (success) reward += 10.0;

        // Penalties
        if (!validAction)              reward -= 0.10;
        if (state.stuckSteps >= STUCK_LIMIT) reward -= 0.25;

        return reward;
    }

    // ------------------------------------------------------------------
    // Task progress (for info dict)
    // ------------------------------------------------------------------

    private double taskProgress(RLNpcEntity agent) {
        if ("farming".equals(state.taskName)) {
            return state.lastInteractValid
                ? 1.0
                : Math.max(0.0, 1.0 - Math.min(distanceToTarget(agent) / 4.0, 1.0));
        }
        return Math.max(0.0, 1.0 - Math.min(distanceToTarget(agent) / 8.0, 1.0));
    }

    // ------------------------------------------------------------------
    // Agent lifecycle
    // ------------------------------------------------------------------

    private RLNpcEntity getOrCreateAgent(ServerLevel level) {
        RLNpcEntity existing = getTrackedAgent(level);
        if (existing != null && existing.isAlive()) return existing;

        RLNpcEntity agent = ModEntities.RL_NPC.get().create(level);
        if (agent == null) throw new IllegalStateException("Failed to create RL NPC entity");

        agent.setCustomName(Component.literal("RL NPC"));
        agent.setCustomNameVisible(true);
        agent.setInvulnerable(true);
        agent.setNoAi(true);
        agent.setSilent(true);
        agent.setPersistenceRequired();

        level.addFreshEntity(agent);
        state.agentUuid = agent.getUUID();
        return agent;
    }

    private RLNpcEntity getTrackedAgent(ServerLevel level) {
        UUID uuid = state.agentUuid;
        if (uuid == null) return null;
        Entity entity = level.getEntity(uuid);
        return (entity instanceof RLNpcEntity npc) ? npc : null;
    }

    private void placeAgent(RLNpcEntity agent, double x, double y, double z, float yaw) {
        agent.moveTo(x, y, z, yaw, 0.0f);
        agent.setDeltaMovement(0.0, 0.0, 0.0);
        agent.fallDistance = 0.0f;
        agent.setYRot(yaw);
        agent.setYHeadRot(yaw);
        agent.yBodyRot = yaw;
        agent.yHeadRot = yaw;
    }

    // ------------------------------------------------------------------
    // Block / marker helpers
    // ------------------------------------------------------------------

    private void clearTaskArtifacts(ServerLevel level) {
        if (state.markerPos != null) {
            level.setBlockAndUpdate(state.markerPos, Blocks.AIR.defaultBlockState());
            state.markerPos = null;
        }
        if (state.farmingCropPos != null) {
            level.setBlockAndUpdate(state.farmingCropPos, Blocks.AIR.defaultBlockState());
            state.farmingCropPos = null;
        }
        if (state.farmingSoilPos != null) {
            level.setBlockAndUpdate(state.farmingSoilPos, Blocks.GRASS_BLOCK.defaultBlockState());
            state.farmingSoilPos = null;
        }
        // Clear navigation obstacles from the previous episode
        for (BlockPos pos : state.obstaclePositions) {
            level.setBlockAndUpdate(pos, Blocks.AIR.defaultBlockState());
        }
        state.obstaclePositions.clear();
    }

    private void clearFarmArea(ServerLevel level, int centerX, int centerZ) {
        int groundY = getReferenceGroundY(level, centerX, centerZ);
        for (int x = centerX - 1; x <= centerX + 1; x++) {
            for (int z = centerZ - 1; z <= centerZ + 1; z++) {
                for (int y = groundY; y <= groundY + 4; y++) {
                    BlockPos pos = new BlockPos(x, y, z);
                    if (!level.getBlockState(pos).is(Blocks.BEDROCK)) {
                        level.setBlockAndUpdate(pos, y == groundY
                                ? Blocks.GRASS_BLOCK.defaultBlockState()
                                : Blocks.AIR.defaultBlockState());
                    }
                }
            }
        }
    }

    private void placeMarker(ServerLevel level) {
        BlockPos markerPos;
        if ("farming".equals(state.taskName) && state.farmingCropPos != null) {
            markerPos = state.farmingCropPos.east();
            level.setBlockAndUpdate(markerPos, Blocks.EMERALD_BLOCK.defaultBlockState());
        } else {
            int markerX  = (int) Math.floor(state.targetX);
            int groundY  = (int) Math.floor(resolveGroundY(level, state.targetX, state.targetZ));
            int markerZ  = (int) Math.floor(state.targetZ);
            markerPos = new BlockPos(markerX, groundY + 1, markerZ);
            level.setBlockAndUpdate(markerPos, Blocks.GOLD_BLOCK.defaultBlockState());
        }
        state.markerPos = markerPos;
    }

    // ------------------------------------------------------------------
    // Y resolution helpers
    // ------------------------------------------------------------------

    private double resolveGroundY(ServerLevel level, double x, double z) {
        BlockPos top = level.getHeightmapPos(
                Heightmap.Types.MOTION_BLOCKING_NO_LEAVES,
                new BlockPos((int) Math.round(x), 0, (int) Math.round(z)));
        return top.getY() - 1.0;
    }

    private double resolveStandY(ServerLevel level, double x, double z) {
        BlockPos top = level.getHeightmapPos(
                Heightmap.Types.MOTION_BLOCKING_NO_LEAVES,
                new BlockPos((int) Math.round(x), 0, (int) Math.round(z)));
        return top.getY();
    }

    private int getReferenceGroundY(ServerLevel level, int x, int z) {
        BlockPos top = level.getHeightmapPos(
                Heightmap.Types.MOTION_BLOCKING_NO_LEAVES, new BlockPos(x, 0, z));
        return top.getY() - 1;
    }

    // ------------------------------------------------------------------
    // Misc
    // ------------------------------------------------------------------

    private double distanceToTarget(RLNpcEntity agent) {
        double dx = state.targetX - agent.getX();
        double dz = state.targetZ - agent.getZ();
        return Math.sqrt(dx * dx + dz * dz);
    }

    private ServerPlayer getObserverPlayer() {
        List<ServerPlayer> players = server.getPlayerList().getPlayers();
        return players.isEmpty() ? null : players.get(0);
    }

    public synchronized void notifyPlayer(String text) {
        ServerPlayer player = getObserverPlayer();
        if (player != null) player.sendSystemMessage(Component.literal(text));
    }

    // ------------------------------------------------------------------
    // JSON helpers
    // ------------------------------------------------------------------

    private Map<String,Object> buildInfo(boolean success, double distance, double progress) {
        Map<String,Object> info = new HashMap<>();
        info.put("task_name",            state.taskName);
        info.put("success",              success);
        info.put("episode_step",         state.episodeStep);
        info.put("distance_to_target",   round(distance));
        info.put("stuck_steps",          state.stuckSteps);
        info.put("task_progress",        round(progress));
        info.put("invalid_action_count", state.invalidActionCount);
        return info;
    }

    private String jsonResponse(double[] obs, double reward, boolean done,
                                boolean truncated, Map<String,Object> info) {
        StringBuilder sb = new StringBuilder("{");
        sb.append("\"obs\":").append(ObservationBuilder.obsToJson(obs)).append(",");
        sb.append("\"reward\":").append(String.format(Locale.US, "%.6f", reward)).append(",");
        sb.append("\"done\":").append(done).append(",");
        sb.append("\"truncated\":").append(truncated).append(",");
        sb.append("\"info\":").append(mapToJson(info));
        sb.append("}");
        return sb.toString();
    }

    private String errorJson(String message) {
        return "{\"error\":\"" + escape(message) + "\"}";
    }

    private String mapToJson(Map<String,Object> map) {
        StringBuilder sb = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<String,Object> e : map.entrySet()) {
            if (!first) sb.append(",");
            first = false;
            sb.append("\"").append(escape(e.getKey())).append("\":");
            Object v = e.getValue();
            if (v instanceof String) sb.append("\"").append(escape((String) v)).append("\"");
            else                     sb.append(v);
        }
        sb.append("}");
        return sb.toString();
    }

    private String escape(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private double round(double v) {
        return Math.round(v * 1000.0) / 1000.0;
    }
}
