package com.aberrada.rlnpc;

import net.minecraft.core.BlockPos;
import net.minecraft.network.chat.Component;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EntityType;
import net.minecraft.world.entity.decoration.ArmorStand;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.CropBlock;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.levelgen.Heightmap;

import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.Random;

public class EnvironmentManager {
    private static final double NAV_SPAWN_X = 0.0;
    private static final double NAV_SPAWN_Z = 0.0;
    private static final float NAV_SPAWN_YAW = -90.0f;

    private static final double NAV_TARGET_X = 8.0;
    private static final double NAV_TARGET_Z = 0.0;

    private static final int FARM_X = 6;
    private static final int FARM_Z = 2;
    private static final int FARM_REF_X = 9;
    private static final int FARM_REF_Z = 2;

    private static final double NAV_SUCCESS_DISTANCE = 1.5;
    private static final double FARM_SUCCESS_DISTANCE = 1.05;
    private static final double STUCK_EPS = 0.01;
    private static final int STUCK_LIMIT = 8;

    private static final double MAX_EPISODE_DISTANCE = 20.0;

    private final Random rng = new Random();

    private final MinecraftServer server;
    private final EpisodeState state;

    public EnvironmentManager(MinecraftServer server) {
        this.server = server;
        this.state = new EpisodeState();
    }

    public synchronized String reset(String taskName) {
        ServerPlayer observer = getObserverPlayer();
        if (observer == null) {
            return errorJson("No player found. Open a singleplayer world first.");
        }

        // Schedule block changes on the main server thread
        server.execute(() -> {
            ServerLevel level = observer.serverLevel();

            // Clear previous farm area BEFORE nulling state fields
            if ("farming".equals(state.taskName) && state.farmingSoilPos != null) {
                clearFarmArea(level, state.farmingSoilPos.getX(), state.farmingSoilPos.getZ());
            }
            clearTaskArtifacts(level);

            state.setTask(taskName);

            if ("farming".equals(state.taskName)) {
                configureFarmingTask(level);
                // Spawn 3–5 blocks from crop, random direction, random yaw
                double spawnAngle = rng.nextDouble() * 2.0 * Math.PI;
                double spawnDist  = 3.0 + rng.nextDouble() * 2.0;
                double spawnX = state.targetX + Math.cos(spawnAngle) * spawnDist;
                double spawnZ = state.targetZ + Math.sin(spawnAngle) * spawnDist;
                double spawnY = resolveStandY(level, spawnX, spawnZ);
                float  spawnYaw = (float)(rng.nextDouble() * 360.0 - 180.0);
                RLNpcEntity agent = getOrCreateAgent(level);
                placeAgent(agent, spawnX, spawnY, spawnZ, spawnYaw);
            } else {
                configureNavigationTask(level);
                // Spawn near origin, random yaw
                double spawnX = (rng.nextDouble() - 0.5) * 2.0;
                double spawnZ = (rng.nextDouble() - 0.5) * 2.0;
                double spawnY = resolveStandY(level, spawnX, spawnZ);
                float  spawnYaw = (float)(rng.nextDouble() * 360.0 - 180.0);
                RLNpcEntity agent = getOrCreateAgent(level);
                placeAgent(agent, spawnX, spawnY, spawnZ, spawnYaw);
            }

            placeMarker(level);
            RLNpcEntity agent = getOrCreateAgent(level);
            double initialDistance = distanceToTarget(agent);
            state.reset(initialDistance);
        });

        // Small sleep to let the main thread finish before we read state
        try { Thread.sleep(50); } catch (InterruptedException ignored) {}

        ServerLevel level = observer.serverLevel();
        RLNpcEntity agent = getOrCreateAgent(level);
        double initialDistance = distanceToTarget(agent);
        double[] obs = ObservationBuilder.build(agent, state);
        Map<String, Object> info = buildInfo(false, initialDistance, taskProgress(agent));
        return jsonResponse(obs, 0.0, false, false, info);
    }

    public synchronized String step(int action) {
        ServerPlayer observer = getObserverPlayer();
        if (observer == null) {
            return errorJson("No player found. Open a singleplayer world first.");
        }

        ServerLevel level = observer.serverLevel();
        RLNpcEntity agent = getOrCreateAgent(level);

        if (state.done) {
            double currentDistance = distanceToTarget(agent);
            double[] obs = ObservationBuilder.build(agent, state);
            Map<String, Object> info = buildInfo(state.success, currentDistance, taskProgress(agent));
            return jsonResponse(obs, 0.0, true, false, info);
        }

        double beforeDistance = distanceToTarget(agent);
        state.lastInteractValid = false;

        boolean validAction;
        if (action == 3) {
            validAction = handleInteract(level, agent);
        } else if (action == 5 && !agent.onGround()) {
            validAction = false;   // penalise mid-air jump attempts
        } else {
            validAction = ActionExecutor.applyMovementAction(agent, action);
        }

        double currentDistance = distanceToTarget(agent);
        double delta = beforeDistance - currentDistance;
        state.episodeStep++;

        if (Math.abs(delta) < STUCK_EPS) {
            state.stuckSteps++;
        } else {
            state.stuckSteps = 0;
        }

        boolean success = computeSuccess(level, agent);
        boolean truncated = state.episodeStep >= state.maxSteps
            || distanceToTarget(agent) > MAX_EPISODE_DISTANCE;
        state.success = success;
        state.done = success || truncated;

        if (!validAction) {
            state.invalidActionCount++;
        }

        double reward = computeReward(agent, action, beforeDistance, currentDistance, validAction, success);
        state.prevDistance = currentDistance;

        double[] obs = ObservationBuilder.build(agent, state);
        Map<String, Object> info = buildInfo(success, currentDistance, taskProgress(agent));
        return jsonResponse(obs, reward, state.done, truncated, info);
    }

    private RLNpcEntity getOrCreateAgent(ServerLevel level) {
    RLNpcEntity existing = getTrackedAgent(level);
    if (existing != null && existing.isAlive()) {
        return existing;
    }

    RLNpcEntity agent = ModEntities.RL_NPC.get().create(level);
        if (agent == null) {
            throw new IllegalStateException("Failed to create RL NPC entity");
        }

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
        if (entity instanceof RLNpcEntity npc) {
            return npc;
        }
        return null;
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

    private void configureNavigationTask(ServerLevel level) {
        double angle = rng.nextDouble() * 2.0 * Math.PI;
        double dist  = 7.0 + rng.nextDouble() * 7.0;          // 7 to 14 blocks
        state.targetX = Math.round(Math.cos(angle) * dist);
        state.targetZ = Math.round(Math.sin(angle) * dist);
        state.targetY = resolveGroundY(level, state.targetX, state.targetZ) + 1.0;
    }

    private void configureFarmingTask(ServerLevel level) {
        int farmX = FARM_X + rng.nextInt(7) - 3;   // ±3 blocks
        int farmZ = FARM_Z + rng.nextInt(7) - 3;

        int groundY = getReferenceGroundY(level, farmX, farmZ);
        BlockPos soilPos = new BlockPos(farmX, groundY, farmZ);
        BlockPos cropPos = soilPos.above();

        level.setBlockAndUpdate(soilPos, Blocks.FARMLAND.defaultBlockState());
        CropBlock wheat = (CropBlock) Blocks.WHEAT;
        level.setBlockAndUpdate(cropPos, wheat.getStateForAge(wheat.getMaxAge()));

        // Target: just west of crop center
        state.targetX = farmX - 0.5;
        state.targetY = cropPos.getY();
        state.targetZ = farmZ + 0.5;

        state.farmingSoilPos = soilPos;
        state.farmingCropPos = cropPos;
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

    private boolean handleInteract(ServerLevel level, RLNpcEntity agent) {
        if (!"farming".equals(state.taskName)) {
            return false;
        }

        BlockPos frontPos = ObservationBuilder.frontCropPos(agent);
        BlockState block = level.getBlockState(frontPos);

        if (block.getBlock() instanceof CropBlock cropBlock && cropBlock.isMaxAge(block)) {
            server.execute(() -> level.destroyBlock(frontPos, true, agent));
            state.lastInteractValid = true;
            return true;
        }
        return false;
    }

    private boolean computeSuccess(ServerLevel level, RLNpcEntity agent) {
        if ("farming".equals(state.taskName)) {
            return state.lastInteractValid
                    && state.farmingCropPos != null
                    && distanceToTarget(agent) <= FARM_SUCCESS_DISTANCE
                    && level.getBlockState(state.farmingCropPos).isAir();
        }
        return distanceToTarget(agent) <= NAV_SUCCESS_DISTANCE;
    }

    private double computeReward(RLNpcEntity agent, int action, double beforeDistance, double currentDistance, boolean validAction, boolean success) {
        double reward = 0.0;
        double progress = beforeDistance - currentDistance;

        reward += 0.10 * progress;
        reward -= 0.01;

        if ("farming".equals(state.taskName)) {
            boolean cropInFront = ObservationBuilder.isMatureCropInFront(agent);
            boolean nearTarget = currentDistance <= 1.10;

            if (nearTarget) {
                reward += 0.04;
            }

            if (cropInFront) {
                reward += 0.25;
            }

            if (beforeDistance <= 1.15 && currentDistance > beforeDistance + 1e-6) {
                reward -= 0.08;
            }

            if (action == 3 && cropInFront) {
                reward += 2.0;
            }

            if (action == 3 && !cropInFront) {
                reward -= 0.03;
            }

            if (state.lastInteractValid) {
                reward += 4.0;
            }
        }

        if (success) {
            reward += 10.0;
        }

        if (!validAction) {
            reward -= 0.10;
        }

        if (state.stuckSteps >= STUCK_LIMIT) {
            reward -= 0.20;
        }

        return reward;
    }

    private double taskProgress(RLNpcEntity  agent) {
        if ("farming".equals(state.taskName)) {
            return state.lastInteractValid ? 1.0 : Math.max(0.0, 1.0 - Math.min(distanceToTarget(agent) / 4.0, 1.0));
        }
        return Math.max(0.0, 1.0 - Math.min(distanceToTarget(agent) / 8.0, 1.0));
    }

    private double distanceToTarget(RLNpcEntity  agent) {
        double dx = state.targetX - agent.getX();
        double dz = state.targetZ - agent.getZ();
        return Math.sqrt(dx * dx + dz * dz);
    }

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
    }

    private void placeMarker(ServerLevel level) {
        BlockPos markerPos;
        if ("farming".equals(state.taskName) && state.farmingCropPos != null) {
            // Place emerald one block east of the crop so it doesn't replace wheat
            markerPos = state.farmingCropPos.east();
            level.setBlockAndUpdate(markerPos, Blocks.EMERALD_BLOCK.defaultBlockState());
        } else {
            int markerX = (int) Math.floor(state.targetX);
            int groundY  = (int) Math.floor(resolveGroundY(level, state.targetX, state.targetZ));
            int markerZ  = (int) Math.floor(state.targetZ);
            markerPos = new BlockPos(markerX, groundY + 1, markerZ);
            level.setBlockAndUpdate(markerPos, Blocks.GOLD_BLOCK.defaultBlockState());
        }
        state.markerPos = markerPos;
    }

    private ServerPlayer getObserverPlayer() {
        List<ServerPlayer> players = server.getPlayerList().getPlayers();
        return players.isEmpty() ? null : players.get(0);
    }

    private Map<String, Object> buildInfo(boolean success, double distance, double progress) {
        Map<String, Object> info = new HashMap<>();
        info.put("task_name", state.taskName);
        info.put("success", success);
        info.put("episode_step", state.episodeStep);
        info.put("distance_to_target", round(distance));
        info.put("stuck_steps", state.stuckSteps);
        info.put("task_progress", round(progress));
        info.put("invalid_action_count", state.invalidActionCount);
        return info;
    }

    private double resolveGroundY(ServerLevel level, double x, double z) {
        BlockPos top = level.getHeightmapPos(
                Heightmap.Types.MOTION_BLOCKING_NO_LEAVES,
                new BlockPos((int) Math.round(x), 0, (int) Math.round(z))
        );
        return top.getY() - 1.0;
    }

    private double resolveStandY(ServerLevel level, double x, double z) {
        BlockPos top = level.getHeightmapPos(
                Heightmap.Types.MOTION_BLOCKING_NO_LEAVES,
                new BlockPos((int) Math.round(x), 0, (int) Math.round(z))
        );
        return top.getY();
    }

    private int getReferenceGroundY(ServerLevel level, int x, int z) {
        BlockPos top = level.getHeightmapPos(
                Heightmap.Types.MOTION_BLOCKING_NO_LEAVES,
                new BlockPos(x, 0, z)
        );
        return top.getY() - 1;
    }

    public synchronized void notifyPlayer(String text) {
        ServerPlayer player = getObserverPlayer();
        if (player != null) {
            player.sendSystemMessage(Component.literal(text));
        }
    }

    private String jsonResponse(double[] obs, double reward, boolean done, boolean truncated, Map<String, Object> info) {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
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

    private String mapToJson(Map<String, Object> map) {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        boolean first = true;
        for (Map.Entry<String, Object> entry : map.entrySet()) {
            if (!first) sb.append(",");
            first = false;
            sb.append("\"").append(escape(entry.getKey())).append("\":");
            Object value = entry.getValue();
            if (value instanceof String) {
                sb.append("\"").append(escape((String) value)).append("\"");
            } else {
                sb.append(value);
            }
        }
        sb.append("}");
        return sb.toString();
    }

    private String escape(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private double round(double v) {
        return Math.round(v * 1000.0) / 1000.0;
    }
}