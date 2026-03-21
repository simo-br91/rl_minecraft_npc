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

        ServerLevel level = observer.serverLevel();
        clearTaskArtifacts(level);
        state.setTask(taskName);

        if ("farming".equals(state.taskName)) {
            configureFarmingTask(level);
        } else {
            configureNavigationTask(level);
        }

        RLNpcEntity agent = getOrCreateAgent(level);

        if ("farming".equals(state.taskName)) {
            // Spawn near the stand position west of the crop and face east toward it.
            double spawnX = state.targetX - 1.4;
            double spawnZ = state.targetZ;
            double spawnY = resolveStandY(level, spawnX, spawnZ);
            placeAgent(agent, spawnX, spawnY, spawnZ, -90.0f);
        } else {
            double spawnY = resolveStandY(level, NAV_SPAWN_X, NAV_SPAWN_Z);
            placeAgent(agent, NAV_SPAWN_X, spawnY, NAV_SPAWN_Z, NAV_SPAWN_YAW);
        }

        placeMarker(level);

        double initialDistance = distanceToTarget(agent);
        state.reset(initialDistance);

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
        boolean truncated = state.episodeStep >= state.maxSteps;
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
        state.targetX = NAV_TARGET_X;
        state.targetY = resolveGroundY(level, NAV_TARGET_X, NAV_TARGET_Z) + 1.0;
        state.targetZ = NAV_TARGET_Z;
    }

    private void configureFarmingTask(ServerLevel level) {
        clearFarmArea(level);

        int groundY = getReferenceGroundY(level, FARM_REF_X, FARM_REF_Z);
        BlockPos soilPos = new BlockPos(FARM_X, groundY, FARM_Z);
        BlockPos cropPos = soilPos.above();

        level.setBlockAndUpdate(soilPos, Blocks.FARMLAND.defaultBlockState());

        CropBlock wheat = (CropBlock) Blocks.WHEAT;
        BlockState matureWheat = wheat.getStateForAge(wheat.getMaxAge());
        level.setBlockAndUpdate(cropPos, matureWheat);

        // Important: target the stand position WEST of the crop, not the crop center.
        state.targetX = FARM_X - 0.5;
        state.targetY = cropPos.getY();
        state.targetZ = FARM_Z + 0.5;

        state.farmingSoilPos = soilPos;
        state.farmingCropPos = cropPos;
    }

    private void clearFarmArea(ServerLevel level) {
        int groundY = getReferenceGroundY(level, FARM_REF_X, FARM_REF_Z);

        for (int x = FARM_X - 1; x <= FARM_X + 1; x++) {
            for (int z = FARM_Z - 1; z <= FARM_Z + 1; z++) {
                for (int y = groundY; y <= groundY + 4; y++) {
                    BlockPos pos = new BlockPos(x, y, z);
                    BlockState current = level.getBlockState(pos);
                    if (!current.is(Blocks.BEDROCK)) {
                        if (y == groundY) {
                            level.setBlockAndUpdate(pos, Blocks.GRASS_BLOCK.defaultBlockState());
                        } else {
                            level.setBlockAndUpdate(pos, Blocks.AIR.defaultBlockState());
                        }
                    }
                }
            }
        }

        state.farmingCropPos = null;
        state.farmingSoilPos = null;
    }

    private boolean handleInteract(ServerLevel level, RLNpcEntity agent) {
        if (!"farming".equals(state.taskName)) {
            return false;
        }

        BlockPos frontPos = ObservationBuilder.frontCropPos(agent);
        BlockState block = level.getBlockState(frontPos);

        if (block.getBlock() instanceof CropBlock cropBlock && cropBlock.isMaxAge(block)) {
            level.destroyBlock(frontPos, true, agent);
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
        if ("farming".equals(state.taskName)) {
            // Marker near the crop, not on the stand target.
            markerPos = new BlockPos(FARM_X, (int) Math.round(state.targetY), FARM_Z + 1);
            level.setBlockAndUpdate(markerPos, Blocks.EMERALD_BLOCK.defaultBlockState());
        } else {
            int markerX = (int) Math.floor(state.targetX);
            int groundY = (int) Math.floor(resolveGroundY(level, state.targetX, state.targetZ));
            int markerZ = (int) Math.floor(state.targetZ);
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