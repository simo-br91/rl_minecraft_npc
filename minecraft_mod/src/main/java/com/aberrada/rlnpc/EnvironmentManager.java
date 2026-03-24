package com.aberrada.rlnpc;

import net.minecraft.core.BlockPos;
import net.minecraft.network.chat.Component;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.Difficulty;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EntityType;
import net.minecraft.world.entity.MobSpawnType;
import net.minecraft.world.entity.monster.Skeleton;
import net.minecraft.world.entity.monster.Zombie;
import net.minecraft.world.level.GameRules;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.CropBlock;
import net.minecraft.world.level.block.state.BlockState;

import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/**
 * Central coordinator for RL environment episodes.
 *
 * Responsibilities:
 *  - Episode lifecycle (reset / step)
 *  - Agent creation and inventory setup
 *  - Delegating to TaskSetup, RewardCalculator, ObservationBuilder
 *  - HTTP JSON serialization
 *  - Night/day control, mob spawning
 *
 * Changes vs previous version:
 *  - Sparse mode: truncates episode when stuckSteps >= SPARSE_STUCK_TRUNCATE_LIMIT
 *    rather than waiting for maxSteps (avoids wasting rollout budget).
 *  - Dead agent entity leak fixed: getOrCreateAgent() removes stale dead
 *    entities before creating a new one.
 *  - Removed unused notifyPlayer() method.
 */
public class EnvironmentManager {

    // ------------------------------------------------------------------
    // Episode constants
    // ------------------------------------------------------------------
    private static final double NAV_SUCCESS_DIST         = 1.5;
    private static final double FARM_SUCCESS_DIST        = 1.05;
    private static final double STUCK_EPS                = 0.01;
    private static final int    STUCK_LIMIT              = 8;
    private static final double MAX_EPISODE_DIST         = 35.0;
    private static final int    DEFAULT_CROP_COUNT       = 5;
    private static final int    MOB_COUNT_PER_EP         = 3;

    /**
     * In sparse-reward mode, auto-truncate if the agent is stuck for this
     * many consecutive steps.  Without this, the agent just runs out maxSteps
     * doing nothing useful, wasting rollout budget.
     */
    private static final int    SPARSE_STUCK_TRUNCATE_LIMIT = 20;

    // ------------------------------------------------------------------
    // Fields
    // ------------------------------------------------------------------
    private final MinecraftServer server;
    private final EpisodeState    state;
    private final Random          rng = new Random();

    public EnvironmentManager(MinecraftServer server) {
        this.server = server;
        this.state  = new EpisodeState();
        server.execute(this::applyWorldSettings);
    }

    // ------------------------------------------------------------------
    // reset()
    // ------------------------------------------------------------------

    public synchronized String reset(String taskName,
                                     boolean sparseReward,
                                     double  minDist,
                                     double  maxDist,
                                     int     numObstacles,
                                     int     numCrops,
                                     boolean fullFarmingCycle) {
        ServerPlayer observer = getObserverPlayer();
        if (observer == null) return errorJson("No player found. Open a singleplayer world first.");

        CompletableFuture<String> future = new CompletableFuture<>();
        server.execute(() -> {
            try {
                ServerLevel level = observer.serverLevel();
                applyWorldSettings();

                // Clean previous episode
                TaskSetup.clearTaskArtifacts(level, state);
                despawnEpisodeMobs(level);

                // Configure task
                state.setTask(taskName);
                state.sparseReward           = sparseReward;
                state.curriculumMinDist      = minDist;
                state.curriculumMaxDist      = maxDist;
                state.curriculumNumObstacles = numObstacles;

                // Create / locate agent
                RLNpcEntity agent = getOrCreateAgent(level);
                InventoryManager.equipAgent(agent, state);

                // Task-specific setup
                double spawnX, spawnZ, spawnY;
                float  spawnYaw = (float)(rng.nextDouble() * 360.0 - 180.0);

                switch (state.taskName) {
                    case "farming" -> {
                        int nc = numCrops > 0 ? numCrops : DEFAULT_CROP_COUNT;
                        TaskSetup.configureFarming(level, state, nc, fullFarmingCycle, rng);
                        spawnX = state.targetX + (rng.nextDouble() - 0.5) * 8.0;
                        spawnZ = state.targetZ + (rng.nextDouble() - 0.5) * 8.0;
                        spawnY = TaskSetup.resolveStandY(level, spawnX, spawnZ);
                    }
                    case "combat" -> {
                        TaskSetup.configureNavigation(level, state, rng);
                        spawnX = (rng.nextDouble() - 0.5) * 2.0;
                        spawnZ = (rng.nextDouble() - 0.5) * 2.0;
                        spawnY = TaskSetup.resolveStandY(level, spawnX, spawnZ);
                        TaskSetup.placeNavigationMarker(level, state);
                        spawnCombatMobs(level, spawnX, spawnZ, MOB_COUNT_PER_EP);
                    }
                    case "multitask" -> {
                        TaskSetup.configureNavigation(level, state, rng);
                        int nc = numCrops > 0 ? numCrops : DEFAULT_CROP_COUNT;
                        TaskSetup.configureFarming(level, state, nc, fullFarmingCycle, rng);
                        state.updateActiveCrop(0, 0);
                        spawnX = (rng.nextDouble() - 0.5) * 3.0;
                        spawnZ = (rng.nextDouble() - 0.5) * 3.0;
                        spawnY = TaskSetup.resolveStandY(level, spawnX, spawnZ);
                        TaskSetup.placeNavigationMarker(level, state);
                        spawnCombatMobs(level, spawnX, spawnZ, 2);
                    }
                    default -> {   // navigation
                        TaskSetup.configureNavigation(level, state, rng);
                        spawnX = (rng.nextDouble() - 0.5) * 2.0;
                        spawnZ = (rng.nextDouble() - 0.5) * 2.0;
                        spawnY = TaskSetup.resolveStandY(level, spawnX, spawnZ);
                        TaskSetup.placeNavigationObstacles(level, state, spawnX, spawnZ, rng);
                        TaskSetup.placeNavigationMarker(level, state);
                    }
                }

                placeAgent(agent, spawnX, spawnY, spawnZ, spawnYaw);

                state.targetPitch  = computeNaturalPitch(agent, state);
                state.currentPitch = state.targetPitch;

                double initDist = distanceToCurrentTarget(agent);
                state.reset(initDist);

                double[]           obs  = ObservationBuilder.build(agent, state);
                Map<String,Object> info = buildInfo(false, initDist, 0.0);
                future.complete(jsonResponse(obs, 0.0, false, false, info));
            } catch (Exception e) {
                RLNpcMod.LOGGER.error("reset() error", e);
                future.complete(errorJson("Reset error: " + e.getMessage()));
            }
        });

        try { return future.get(12, TimeUnit.SECONDS); }
        catch (Exception e) { return errorJson("Reset timed out: " + e.getMessage()); }
    }

    // ------------------------------------------------------------------
    // step()
    // ------------------------------------------------------------------

    public synchronized String step(int action) {
        ServerPlayer observer = getObserverPlayer();
        if (observer == null) return errorJson("No player found.");

        CompletableFuture<String> future = new CompletableFuture<>();
        server.execute(() -> {
            try {
                ServerLevel level = observer.serverLevel();
                RLNpcEntity agent = getOrCreateAgent(level);

                if (state.done) {
                    double cd = distanceToCurrentTarget(agent);
                    future.complete(jsonResponse(ObservationBuilder.build(agent, state),
                            0.0, true, false,
                            buildInfo(state.success, cd, taskProgress(agent))));
                    return;
                }

                double beforeDist = distanceToCurrentTarget(agent);
                state.lastInteractValid  = false;
                state.lastJumpedObstacle = false;
                state.lastAttackValid    = false;
                state.lastEatValid       = false;
                state.lastSprinting      = false;
                state.mobsKilled         = 0;
                state.timesHit           = 0;

                updateHealthAndFood(agent);

                boolean validAction;
                if (action == 3) {
                    validAction = handleInteract(level, agent);
                } else if (action == 10) {
                    validAction = ActionExecutor.attack(agent, state);
                } else if (action == 11) {
                    validAction = InventoryManager.eatFood(agent, state);
                } else if (action == 12) {
                    validAction = handleSwitchItem(agent);
                } else {
                    validAction = ActionExecutor.applyAction(agent, action, state);
                }

                state.targetPitch = computeNaturalPitch(agent, state);
                ActionExecutor.updatePitch(agent, state);

                double afterDist = distanceToCurrentTarget(agent);
                double delta     = beforeDist - afterDist;
                state.episodeStep++;

                if (Math.abs(delta) < STUCK_EPS) state.stuckSteps++;
                else                              state.stuckSteps = 0;

                boolean success   = computeSuccess(level, agent);
                boolean truncated = isTruncated(agent);

                state.success = success;
                state.done    = success || truncated;
                if (!validAction) state.invalidActionCount++;

                state.health = agent.getHealth();
                state.isDead = !agent.isAlive() || agent.getHealth() <= 0;

                if ("farming".equals(state.taskName) || "multitask".equals(state.taskName)) {
                    state.updateActiveCrop(agent.getX(), agent.getZ());
                }

                boolean cropInFront = ObservationBuilder.isMatureCropInFront(agent);
                double reward = RewardCalculator.compute(
                        agent, state, action, beforeDist, afterDist,
                        validAction, success, cropInFront);
                state.prevDistance = afterDist;

                double[]           obs  = ObservationBuilder.build(agent, state);
                Map<String,Object> info = buildInfo(success, afterDist, taskProgress(agent));
                future.complete(jsonResponse(obs, reward, state.done, truncated, info));

            } catch (Exception e) {
                RLNpcMod.LOGGER.error("step() error", e);
                future.complete(errorJson("Step error: " + e.getMessage()));
            }
        });

        try { return future.get(6, TimeUnit.SECONDS); }
        catch (Exception e) { return errorJson("Step timed out: " + e.getMessage()); }
    }

    // ------------------------------------------------------------------
    // Truncation — unified, includes sparse-mode stuck auto-truncation
    // ------------------------------------------------------------------

    private boolean isTruncated(RLNpcEntity agent) {
        if (state.episodeStep >= state.maxSteps)           return true;
        if (distanceToCurrentTarget(agent) > MAX_EPISODE_DIST) return true;
        if (state.isDead)                                  return true;

        // In sparse mode: auto-truncate after prolonged stuckness so the
        // agent doesn't waste the entire episode budget doing nothing.
        if (state.sparseReward && state.stuckSteps >= SPARSE_STUCK_TRUNCATE_LIMIT) {
            return true;
        }
        return false;
    }

    // ------------------------------------------------------------------
    // Interact handler (farming + bonemeal + tilling)
    // ------------------------------------------------------------------

    private boolean handleInteract(ServerLevel level, RLNpcEntity agent) {
        if (!"farming".equals(state.taskName) && !"multitask".equals(state.taskName))
            return false;

        BlockPos frontPos = ObservationBuilder.frontCropPos(agent);
        BlockState block  = level.getBlockState(frontPos);

        if (block.getBlock() instanceof CropBlock cropBlock && cropBlock.isMaxAge(block)) {
            level.destroyBlock(frontPos, true, agent);
            markCropHarvested(frontPos);
            state.lastInteractValid = true;
            if (state.fullFarmingCycle) {
                BlockPos soilPos = frontPos.below();
                if (level.getBlockState(soilPos).getBlock()
                        instanceof net.minecraft.world.level.block.FarmBlock) {
                    CropBlock wheat = (CropBlock) Blocks.WHEAT;
                    level.setBlockAndUpdate(frontPos, wheat.getStateForAge(0));
                    updateCropAfterReplant(frontPos);
                }
            }
            return true;
        }

        // Try bonemeal on non-mature crop
        if (state.fullFarmingCycle
                && block.getBlock() instanceof CropBlock cb && !cb.isMaxAge(block)) {
            if (InventoryManager.hasBonemeal(agent) && InventoryManager.consumeBonemeal(agent)) {
                int idx = getCropIndexForPos(frontPos);
                if (idx >= 0) TaskSetup.applyBonemeal(level, state, idx);
                return true;
            }
        }
        return false;
    }

    private void markCropHarvested(BlockPos cropPos) {
        for (int i = 0; i < state.farmingCropPositions.size(); i++) {
            if (state.farmingCropPositions.get(i).equals(cropPos)) {
                if (i < state.cropHarvested.size()) {
                    state.cropHarvested.set(i, true);
                    state.cropsHarvested++;
                }
                return;
            }
        }
    }

    private int getCropIndexForPos(BlockPos pos) {
        for (int i = 0; i < state.farmingCropPositions.size(); i++) {
            if (state.farmingCropPositions.get(i).equals(pos)) return i;
        }
        return -1;
    }

    private void updateCropAfterReplant(BlockPos pos) {
        int idx = getCropIndexForPos(pos);
        if (idx >= 0 && idx < state.cropGrowthStages.size()) {
            state.cropGrowthStages.set(idx, 0);
            state.cropHarvested.set(idx, false);
            state.cropsHarvested = Math.max(0, state.cropsHarvested - 1);
        }
    }

    // ------------------------------------------------------------------
    // Inventory switch
    // ------------------------------------------------------------------

    private boolean handleSwitchItem(RLNpcEntity agent) {
        state.activeSlot = (state.activeSlot + 1) % 5;
        InventoryManager.syncMainHandFromSlot(agent, state);
        return true;
    }

    // ------------------------------------------------------------------
    // Health and food simulation
    // ------------------------------------------------------------------

    private void updateHealthAndFood(RLNpcEntity agent) {
        state.health = agent.getHealth();

        if (state.lastSprinting && state.foodLevel > 0) {
            state.foodLevel = Math.max(0, state.foodLevel - 1);
        }
        if (state.foodLevel == 0 && agent.getHealth() > 1.0f) {
            agent.hurt(agent.damageSources().starve(), 0.5f);
        }
    }

    // ------------------------------------------------------------------
    // Combat mob spawning
    // ------------------------------------------------------------------

    private void spawnCombatMobs(ServerLevel level, double cx, double cz, int count) {
        state.hostileMobUuids.clear();
        for (int i = 0; i < count; i++) {
            double angle = rng.nextDouble() * 2 * Math.PI;
            double dist  = 6.0 + rng.nextDouble() * 4.0;
            double mx = cx + Math.cos(angle) * dist;
            double mz = cz + Math.sin(angle) * dist;
            double my = TaskSetup.resolveStandY(level, mx, mz);

            net.minecraft.world.entity.monster.Monster mob =
                    (i % 2 == 0) ? new Zombie(level) : new Skeleton(level);
            mob.moveTo(mx, my, mz, (float)(rng.nextDouble() * 360), 0);
            mob.finalizeSpawn(level,
                    level.getCurrentDifficultyAt(BlockPos.containing(mx, my, mz)),
                    MobSpawnType.MOB_SUMMONED, null, null);
            level.addFreshEntity(mob);
            state.hostileMobUuids.add(mob.getUUID());
        }
    }

    private void despawnEpisodeMobs(ServerLevel level) {
        for (UUID uuid : state.hostileMobUuids) {
            Entity e = level.getEntity(uuid);
            if (e != null && e.isAlive()) e.remove(Entity.RemovalReason.DISCARDED);
        }
        state.hostileMobUuids.clear();
    }

    // ------------------------------------------------------------------
    // Success computation
    // ------------------------------------------------------------------

    private boolean computeSuccess(ServerLevel level, RLNpcEntity agent) {
        return switch (state.taskName) {
            case "farming"   -> state.allCropsHarvested();
            case "combat"    -> state.hostileMobUuids.stream()
                    .allMatch(uuid -> {
                        Entity e = level.getEntity(uuid);
                        return e == null || !e.isAlive();
                    });
            case "multitask" -> state.allCropsHarvested()
                    && distanceToCurrentTarget(agent) <= NAV_SUCCESS_DIST;
            default          -> distanceToCurrentTarget(agent) <= NAV_SUCCESS_DIST;
        };
    }

    // ------------------------------------------------------------------
    // Task progress
    // ------------------------------------------------------------------

    private double taskProgress(RLNpcEntity agent) {
        return switch (state.taskName) {
            case "farming", "multitask" -> {
                if (state.totalCrops == 0) yield 0.0;
                yield (double) state.cropsHarvested / state.totalCrops;
            }
            case "combat" -> {
                if (state.hostileMobUuids.isEmpty()) yield 1.0;
                ServerPlayer obs = getObserverPlayer();
                long aliveMobs = state.hostileMobUuids.stream()
                        .filter(uuid -> {
                            if (obs == null) return false;
                            Entity e = obs.serverLevel().getEntity(uuid);
                            return e != null && e.isAlive();
                        }).count();
                yield 1.0 - (double) aliveMobs / state.hostileMobUuids.size();
            }
            default -> {
                double dist = distanceToCurrentTarget(agent);
                double maxD = Math.max(state.curriculumMaxDist, 10.0);
                yield Math.max(0.0, 1.0 - dist / maxD);
            }
        };
    }

    // ------------------------------------------------------------------
    // Pitch helper
    // ------------------------------------------------------------------

    private float computeNaturalPitch(RLNpcEntity agent, EpisodeState state) {
        double dist = distanceToCurrentTarget(agent);
        if ("farming".equals(state.taskName) && dist < 2.0) return 30.0f;
        if (dist < 3.0) return 15.0f;
        return 0.0f;
    }

    // ------------------------------------------------------------------
    // World settings
    // ------------------------------------------------------------------

    private void applyWorldSettings() {
        ServerPlayer p = getObserverPlayer();
        if (p == null) return;
        ServerLevel level = p.serverLevel();
        // Always night so mobs don't burn from sunlight
        level.setDayTime(18000L);
        if (level.getDifficulty() == Difficulty.PEACEFUL) {
            server.setDifficulty(Difficulty.NORMAL, true);
        }
        // Disable natural mob spawning (we spawn manually)
        level.getGameRules().getRule(GameRules.RULE_DOMOBSPAWNING).set(false, server);
        // Keep inventory on death — agent should respawn with gear
        level.getGameRules().getRule(GameRules.RULE_KEEPINVENTORY).set(true, server);
        // Disable daylight cycle so it stays night permanently
        level.getGameRules().getRule(GameRules.RULE_DAYLIGHT).set(false, server);
    }

    // ------------------------------------------------------------------
    // Agent lifecycle
    // ------------------------------------------------------------------

    private RLNpcEntity getOrCreateAgent(ServerLevel level) {
        if (state.agentUuid != null) {
            Entity e = level.getEntity(state.agentUuid);
            if (e instanceof RLNpcEntity npc && npc.isAlive()) {
                return npc;
            }
            // Entity is dead or missing — remove it to prevent accumulation
            if (e != null) {
                e.remove(Entity.RemovalReason.DISCARDED);
            }
            state.agentUuid = null;
        }
        return spawnFreshAgent(level);
    }

    private RLNpcEntity spawnFreshAgent(ServerLevel level) {
        RLNpcEntity agent = ModEntities.RL_NPC.get().create(level);
        if (agent == null) throw new IllegalStateException("Failed to create RLNpcEntity");
        agent.setCustomName(Component.literal("RL Agent"));
        agent.setCustomNameVisible(true);
        agent.setInvulnerable(false);
        agent.setNoAi(true);
        agent.setSilent(true);
        agent.setPersistenceRequired();
        agent.getAttribute(net.minecraft.world.entity.ai.attributes.Attributes.MAX_HEALTH)
             .setBaseValue(20.0);
        agent.setHealth(20.0f);
        level.addFreshEntity(agent);
        state.agentUuid = agent.getUUID();
        return agent;
    }

    private void placeAgent(RLNpcEntity agent, double x, double y, double z, float yaw) {
        agent.moveTo(x, y, z, yaw, 0.0f);
        agent.setDeltaMovement(0.0, 0.0, 0.0);
        agent.fallDistance = 0.0f;
        agent.setHealth(20.0f);
        ActionExecutor.syncRotations(agent, yaw, 0.0f);
        state.health    = 20.0f;
        state.foodLevel  = 20;
        state.saturation = 5.0f;
    }

    // ------------------------------------------------------------------
    // Distance helpers
    // ------------------------------------------------------------------

    private double distanceToCurrentTarget(RLNpcEntity agent) {
        double dx = state.targetX - agent.getX();
        double dz = state.targetZ - agent.getZ();
        return Math.sqrt(dx * dx + dz * dz);
    }

    // ------------------------------------------------------------------
    // Misc
    // ------------------------------------------------------------------

    private ServerPlayer getObserverPlayer() {
        List<ServerPlayer> players = server.getPlayerList().getPlayers();
        return players.isEmpty() ? null : players.get(0);
    }

    // ------------------------------------------------------------------
    // JSON helpers
    // ------------------------------------------------------------------

    private Map<String,Object> buildInfo(boolean success, double dist, double progress) {
        Map<String,Object> info = new HashMap<>();
        info.put("task_name",            state.taskName);
        info.put("success",              success);
        info.put("episode_step",         state.episodeStep);
        info.put("distance_to_target",   round(dist));
        info.put("stuck_steps",          state.stuckSteps);
        info.put("task_progress",        round(progress));
        info.put("invalid_action_count", state.invalidActionCount);
        info.put("sparse_reward",        state.sparseReward);
        info.put("health",               round(state.health));
        info.put("food_level",           state.foodLevel);
        info.put("crops_harvested",      state.cropsHarvested);
        info.put("total_crops",          state.totalCrops);
        info.put("mobs_killed",          state.mobsKilled);
        info.put("active_slot",          state.activeSlot);
        info.put("active_item",          InventoryManager.getActiveItemName(state));
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
