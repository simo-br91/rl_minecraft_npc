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
import net.minecraft.world.level.GameRules;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.CropBlock;
import net.minecraft.world.level.block.state.BlockState;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
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
 * Fixes vs previous version:
 *  - Tick alignment (Bug 2.13): all world/entity mutations are submitted to the
 *    server tick thread via server.execute() + CompletableFuture. The reset()
 *    and step() methods block until the tick has completed before returning.
 *    This was already partially done; the fix ensures the CompletableFuture is
 *    always resolved on the game thread and never on the HTTP thread.
 *  - Seed forwarding (Req 3.7): Java RNG is seeded with the value sent by Python
 *    so episodes are reproducible when a seed is provided.
 *  - Observer independence (Issue 7.4): world operations now use the server's
 *    overworld directly rather than the first player's level, so the system
 *    works correctly in multi-player sessions and headless servers.
 *  - Stuck-near-target (Issue 5.2): a drift penalty is applied when the agent
 *    is within 1.5 blocks but moves away, preventing indefinite small rewards.
 *  - Jump bonus reduced (Issue 5.3): moved from RewardCalculator constant; now
 *    0.20 to be proportional to typical progress rewards.
 *  - Trajectory logging (Issue 7.7): (obs, action, reward, done) tuples are
 *    written to python_rl/logs/trajectory.jsonl when TRAJECTORY_LOGGING=true.
 */
public class EnvironmentManager {

    // ------------------------------------------------------------------
    // Episode constants
    // ------------------------------------------------------------------
    private static final double NAV_SUCCESS_DIST             = 1.5;
    private static final double STUCK_EPS                    = 0.01;
    private static final int    STUCK_LIMIT                  = 8;
    private static final double MAX_EPISODE_DIST             = 35.0;
    private static final int    DEFAULT_CROP_COUNT           = 5;
    private static final int    MOB_COUNT_PER_EP             = 3;
    private static final int    SPARSE_STUCK_TRUNCATE_LIMIT  = 20;

    // Trajectory logging — set to "true" via system property or env var to enable
    private static final boolean TRAJECTORY_LOGGING =
            Boolean.parseBoolean(System.getProperty("rlnpc.trajectory", "false"));
    private static final String TRAJECTORY_FILE = "python_rl/logs/trajectory.jsonl";

    // ------------------------------------------------------------------
    // Fields
    // ------------------------------------------------------------------
    private final MinecraftServer server;
    private final EpisodeState    state;
    private final Random          rng = new Random();
    private PrintWriter           trajectoryWriter = null;

    public EnvironmentManager(MinecraftServer server) {
        this.server = server;
        this.state  = new EpisodeState();
        server.execute(this::applyWorldSettings);
        if (TRAJECTORY_LOGGING) initTrajectoryLog();
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
                                     boolean fullFarmCycle,
                                     int     seed,
                                     int     numMobs,
                                     double  mobDistMin,
                                     double  mobDistMax) {
        // FIX 3.7: seed the Java RNG so episodes are reproducible
        if (seed >= 0) rng.setSeed(seed);

        CompletableFuture<String> future = new CompletableFuture<>();
        server.execute(() -> {
            try {
                // FIX 7.4: use overworld directly instead of player's level
                ServerLevel level = getOverworld();
                if (level == null) {
                    future.complete(errorJson("Server level not available."));
                    return;
                }

                applyWorldSettings();

                TaskSetup.clearTaskArtifacts(level, state);
                despawnEpisodeMobs(level);

                state.setTask(taskName);
                state.sparseReward           = sparseReward;
                state.curriculumMinDist      = minDist;
                state.curriculumMaxDist      = maxDist;
                state.curriculumNumObstacles = numObstacles;

                RLNpcEntity agent = getOrCreateAgent(level);
                InventoryManager.equipAgent(agent, state);

                double spawnX, spawnZ, spawnY;
                float  spawnYaw = (float)(rng.nextDouble() * 360.0 - 180.0);

                switch (state.taskName) {
                    case "farming" -> {
                        int nc = numCrops > 0 ? numCrops : DEFAULT_CROP_COUNT;
                        TaskSetup.configureFarming(level, state, nc, fullFarmCycle, rng);
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
                        // Fix 4.5: use curriculum-controlled mob count/distance (-1 = defaults)
                        int   cCount   = numMobs    > 0   ? numMobs    : MOB_COUNT_PER_EP;
                        double cDistMin = mobDistMin >= 0.0 ? mobDistMin : 6.0;
                        double cDistMax = mobDistMax >= 0.0 ? mobDistMax : 10.0;
                        spawnCombatMobs(level, spawnX, spawnZ, cCount, cDistMin, cDistMax);
                    }
                    case "multitask" -> {
                        TaskSetup.configureNavigation(level, state, rng);
                        // FIX 4.1: save the nav waypoint BEFORE configureFarming
                        // overwrites state.targetX/Z with the first crop position.
                        // The gold-block marker and the success condition both use
                        // navTargetX/Z so they remain consistent throughout.
                        state.navTargetX = state.targetX;
                        state.navTargetZ = state.targetZ;
                        int nc = numCrops > 0 ? numCrops : DEFAULT_CROP_COUNT;
                        TaskSetup.configureFarming(level, state, nc, fullFarmCycle, rng);
                        state.updateActiveCrop(0, 0);
                        spawnX = (rng.nextDouble() - 0.5) * 3.0;
                        spawnZ = (rng.nextDouble() - 0.5) * 3.0;
                        spawnY = TaskSetup.resolveStandY(level, spawnX, spawnZ);
                        TaskSetup.placeNavigationMarker(level, state);
                        spawnCombatMobs(level, spawnX, spawnZ, 2, 6.0, 10.0);
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
        CompletableFuture<String> future = new CompletableFuture<>();
        server.execute(() -> {
            try {
                // FIX 7.4: use overworld directly
                ServerLevel level = getOverworld();
                if (level == null) {
                    future.complete(errorJson("Server level not available."));
                    return;
                }

                RLNpcEntity agent = getOrCreateAgent(level);

                if (state.done) {
                    double cd = distanceToCurrentTarget(agent);
                    future.complete(jsonResponse(ObservationBuilder.build(agent, state),
                            0.0, true, false,
                            buildInfo(state.success, cd, taskProgress(agent, level))));
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

                // Fix 4.3: track nearest mob distance for combat proximity reward
                if ("combat".equals(state.taskName) || "multitask".equals(state.taskName)) {
                    state.nearestMobDist = computeNearestMobDist(level, agent);
                }

                if ("farming".equals(state.taskName) || "multitask".equals(state.taskName)) {
                    state.updateActiveCrop(agent.getX(), agent.getZ());
                }

                boolean cropInFront = ObservationBuilder.isMatureCropInFront(agent);
                double reward = RewardCalculator.compute(
                        agent, state, action, beforeDist, afterDist,
                        validAction, success, cropInFront);
                state.prevDistance = afterDist;

                double[]           obs  = ObservationBuilder.build(agent, state);
                Map<String,Object> info = buildInfo(success, afterDist, taskProgress(agent, level));

                // FIX 7.7: trajectory logging
                if (TRAJECTORY_LOGGING) {
                    logTrajectory(obs, action, reward, state.done, info);
                }

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
    // Truncation
    // ------------------------------------------------------------------

    private boolean isTruncated(RLNpcEntity agent) {
        if (state.episodeStep >= state.maxSteps)               return true;
        if (distanceToCurrentTarget(agent) > MAX_EPISODE_DIST) return true;
        if (state.isDead)                                       return true;
        if (state.sparseReward && state.stuckSteps >= SPARSE_STUCK_TRUNCATE_LIMIT) return true;
        return false;
    }

    // ------------------------------------------------------------------
    // Interact handler
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
    // Health / food simulation
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

    private void spawnCombatMobs(ServerLevel level, double cx, double cz, int count,
                                     double distMin, double distMax) {
        state.hostileMobUuids.clear();
        for (int i = 0; i < count; i++) {
            double angle = rng.nextDouble() * 2 * Math.PI;
            double dist  = distMin + rng.nextDouble() * Math.max(0.0, distMax - distMin);
            double mx = cx + Math.cos(angle) * dist;
            double mz = cz + Math.sin(angle) * dist;
            double my = TaskSetup.resolveStandY(level, mx, mz);

            net.minecraft.world.entity.monster.Monster mob = (i % 2 == 0)
                    ? net.minecraft.world.entity.EntityType.ZOMBIE.create(level)
                    : net.minecraft.world.entity.EntityType.SKELETON.create(level);
            if (mob == null) continue;   // EntityType.create() returns null on failure
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
            // FIX 4.1: use navTargetX/Z (the gold-block position) so the visual
            // goal and the success condition are the same location.  Previously
            // this used distanceToCurrentTarget() which, by episode end, pointed
            // at the last harvested crop — meaning the agent got success credit
            // immediately on the last harvest without reaching the gold block.
            case "multitask" -> state.allCropsHarvested()
                    && distanceToNavTarget(agent) <= NAV_SUCCESS_DIST;
            default          -> distanceToCurrentTarget(agent) <= NAV_SUCCESS_DIST;
        };
    }

    /**
     * Returns true when the block directly in front of the agent is a
     * CropBlock that has NOT yet reached max age (i.e., not harvestable).
     * Used by the action mask to allow bonemeal application in full-cycle
     * farming mode. (Fix 4.2a)
     */
    private boolean isImmatureCropAhead(RLNpcEntity agent) {
        ServerLevel level = getOverworld();
        if (level == null) return false;
        BlockPos pos = ObservationBuilder.frontCropPos(agent);
        BlockState bs = level.getBlockState(pos);
        return bs.getBlock() instanceof net.minecraft.world.level.block.CropBlock cb
                && !cb.isMaxAge(bs);
    }

    /** Distance to the permanent nav waypoint (gold block). Used for multitask. */
    private double distanceToNavTarget(RLNpcEntity agent) {
        double dx = state.navTargetX - agent.getX();
        double dz = state.navTargetZ - agent.getZ();
        return Math.sqrt(dx * dx + dz * dz);
    }

    // ------------------------------------------------------------------
    // Task progress
    // ------------------------------------------------------------------

    private double taskProgress(RLNpcEntity agent, ServerLevel level) {
        return switch (state.taskName) {
            case "farming", "multitask" -> {
                if (state.totalCrops == 0) yield 0.0;
                yield (double) state.cropsHarvested / state.totalCrops;
            }
            case "combat" -> {
                if (state.hostileMobUuids.isEmpty()) yield 1.0;
                long aliveMobs = state.hostileMobUuids.stream()
                        .filter(uuid -> {
                            Entity e = level.getEntity(uuid);
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
    // World settings — FIX 7.4: uses getOverworld(), not player level
    // ------------------------------------------------------------------

    private void applyWorldSettings() {
        ServerLevel level = getOverworld();
        if (level == null) return;
        level.setDayTime(18000L);
        if (level.getDifficulty() == Difficulty.PEACEFUL) {
            server.setDifficulty(Difficulty.NORMAL, true);
        }
        level.getGameRules().getRule(GameRules.RULE_DOMOBSPAWNING).set(false, server);
        level.getGameRules().getRule(GameRules.RULE_KEEPINVENTORY).set(true, server);
        level.getGameRules().getRule(GameRules.RULE_DAYLIGHT).set(false, server);
    }

    // ------------------------------------------------------------------
    // FIX 7.4: world access independent of any player being present
    // ------------------------------------------------------------------

    /**
     * Returns the server's overworld level directly.
     * This does NOT depend on a player being logged in, making the system
     * usable on dedicated servers and in multi-player sessions.
     */
    private ServerLevel getOverworld() {
        return server.getLevel(net.minecraft.world.level.Level.OVERWORLD);
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
            if (e != null) e.remove(Entity.RemovalReason.DISCARDED);
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
    // Action mask (Issue 6.3 — MaskablePPO support)
    // ------------------------------------------------------------------

    /**
     * Returns a 13-element boolean array indicating which actions are valid
     * in the current episode state.  The Python side uses this array with
     * sb3-contrib MaskablePPO to prevent the policy from sampling actions
     * that are known to be invalid, speeding up convergence.
     *
     * Mask logic mirrors ActionExecutor preconditions exactly:
     *   0  forward          always valid
     *   1  turn_left        always valid
     *   2  turn_right       always valid
     *   3  interact         valid if farming/multitask AND mature crop in front
     *   4  no_op            always valid
     *   5  jump             valid if 1-block wall ahead AND clear above
     *   6  sprint_forward   always valid
     *   7  move_backward    always valid
     *   8  strafe_left      always valid
     *   9  strafe_right     always valid
     *  10  attack           valid if hostile mob within attack range
     *  11  eat              valid if food < 20 AND food slot not empty
     *  12  switch_item      always valid
     */
    public synchronized String actionMasks() {
        CompletableFuture<String> future = new CompletableFuture<>();
        server.execute(() -> {
            try {
                ServerLevel level = getOverworld();
                if (level == null) {
                    future.complete("{\"masks\":[1,1,1,1,1,1,1,1,1,1,1,1,1]}");
                    return;
                }
                RLNpcEntity agent = getOrCreateAgent(level);
                boolean[] mask = computeActionMask(agent);
                StringBuilder sb = new StringBuilder("{\"masks\":[");
                for (int i = 0; i < mask.length; i++) {
                    if (i > 0) sb.append(",");
                    sb.append(mask[i] ? 1 : 0);
                }
                sb.append("]}");
                future.complete(sb.toString());
            } catch (Exception e) {
                // Fall back to all-valid mask on error
                future.complete("{\"masks\":[1,1,1,1,1,1,1,1,1,1,1,1,1]}");
            }
        });
        try { return future.get(3, TimeUnit.SECONDS); }
        catch (Exception e) { return "{\"masks\":[1,1,1,1,1,1,1,1,1,1,1,1,1]}"; }
    }

    private boolean[] computeActionMask(RLNpcEntity agent) {
        boolean[] mask = new boolean[13];
        java.util.Arrays.fill(mask, true);   // default: all valid

        // Action 3 — interact:
        //   • always valid when facing a mature (harvestable) crop
        //   • FIX 4.2a: also valid in full-cycle farming when facing an
        //     IMMATURE crop and the agent has bonemeal.  The previous mask
        //     blocked this entirely, making bonemeal application unlearnable.
        boolean farmingTask = "farming".equals(state.taskName)
                           || "multitask".equals(state.taskName);
        boolean matureCropAhead = farmingTask && ObservationBuilder.isMatureCropInFront(agent);
        boolean bonemealApplicable = farmingTask && state.fullFarmingCycle
                && isImmatureCropAhead(agent)
                && InventoryManager.hasBonemeal(agent);
        mask[3] = matureCropAhead || bonemealApplicable;

        // Action 5 — jump: only valid when a 1-block wall is ahead AND passable above
        mask[5] = ObservationBuilder.is1BlockObstacleAhead(agent);

        // Action 10 — attack: only valid when a mob is within attack range
        net.minecraft.world.phys.AABB box = agent.getBoundingBox()
                .inflate(ActionExecutor.ATTACK_RANGE);
        java.util.List<net.minecraft.world.entity.monster.Monster> mobs =
                agent.level().getEntitiesOfClass(
                        net.minecraft.world.entity.monster.Monster.class, box,
                        m -> m.isAlive() && !m.isSpectator());
        mask[10] = !mobs.isEmpty();

        // Action 11 — eat: only valid when hungry, food slot is not empty,
        // AND the food slot is currently active.  FIX 4.2b: InventoryManager
        // .eatFood() eats from the active item; if the sword is held the eat
        // action returns false (invalid), so the mask must require the correct
        // active slot to prevent the agent from collecting the invalid penalty
        // for eating while holding the wrong item.
        mask[11] = state.foodLevel < 20
                && state.activeSlot == EpisodeState.SLOT_FOOD
                && !agent.getInventory()
                         .getItem(InventoryManager.SLOT_FOOD).isEmpty();

        return mask;
    }

    // ------------------------------------------------------------------
    // Distance helpers
    // ------------------------------------------------------------------

    /**
     * Euclidean distance from the agent to the nearest alive hostile mob.
     * Returns {@link Double#MAX_VALUE} when no mobs are tracked.
     * (Fix 4.3 — used by RewardCalculator for combat proximity reward)
     */
    private double computeNearestMobDist(ServerLevel level, RLNpcEntity agent) {
        double best = Double.MAX_VALUE;
        for (UUID uuid : state.hostileMobUuids) {
            Entity e = level.getEntity(uuid);
            if (e != null && e.isAlive()) {
                double d = agent.distanceTo(e);
                if (d < best) best = d;
            }
        }
        return best;
    }

    private double distanceToCurrentTarget(RLNpcEntity agent) {
        double dx = state.targetX - agent.getX();
        double dz = state.targetZ - agent.getZ();
        return Math.sqrt(dx * dx + dz * dz);
    }

    // ------------------------------------------------------------------
    // Trajectory logging (Issue 7.7)
    // ------------------------------------------------------------------

    private void initTrajectoryLog() {
        try {
            Path logDir = Paths.get("python_rl/logs");
            Files.createDirectories(logDir);
            trajectoryWriter = new PrintWriter(new FileWriter(TRAJECTORY_FILE, true));
            RLNpcMod.LOGGER.info("Trajectory logging enabled → {}", TRAJECTORY_FILE);
        } catch (IOException e) {
            RLNpcMod.LOGGER.warn("Could not open trajectory log: {}", e.getMessage());
        }
    }

    private void logTrajectory(double[] obs, int action, double reward,
                                boolean done, Map<String,Object> info) {
        if (trajectoryWriter == null) return;
        try {
            StringBuilder sb = new StringBuilder("{");
            sb.append("\"obs\":").append(ObservationBuilder.obsToJson(obs)).append(",");
            sb.append("\"action\":").append(action).append(",");
            sb.append(String.format(Locale.US, "\"reward\":%.6f,", reward));
            sb.append("\"done\":").append(done).append(",");
            sb.append("\"info\":").append(mapToJson(info));
            sb.append("}");
            trajectoryWriter.println(sb);
            trajectoryWriter.flush();
        } catch (Exception e) {
            RLNpcMod.LOGGER.warn("Trajectory log write error: {}", e.getMessage());
        }
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
