package com.aberrada.rlnpc;

import net.minecraft.core.BlockPos;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Per-episode mutable state shared between EnvironmentManager, ActionExecutor,
 * ObservationBuilder, and RewardCalculator.
 *
 * Extended to support:
 *  - Health & food tracking
 *  - Inventory slots (sword, food, seeds, tools)
 *  - Combat (zombies / skeletons)
 *  - Full farming cycle (till → plant → grow → harvest)
 *  - Multiple crop plots per episode
 *  - Walk vs run speed
 *  - Pitch (vertical look angle) for human-like movement
 */
public class EpisodeState {

    // ------------------------------------------------------------------
    // Episode bookkeeping
    // ------------------------------------------------------------------
    public int     episodeStep        = 0;
    public int     maxSteps           = 150;
    public double  prevDistance       = 0.0;
    public int     stuckSteps         = 0;
    public int     invalidActionCount = 0;
    public boolean done               = false;
    public boolean success            = false;

    // ------------------------------------------------------------------
    // Task identity
    // ------------------------------------------------------------------
    public String  taskName = "navigation";
    public double  taskId   = 0.0;   // 0=nav, 1=farm, 2=combat, 3=multitask

    // ------------------------------------------------------------------
    // Navigation target
    // ------------------------------------------------------------------
    public double  targetX = 8.0;
    public double  targetY = 4.0;
    public double  targetZ = 0.0;

    /**
     * Permanent navigation waypoint for the multitask episode.
     *
     * In multitask mode, {@code targetX/Z} is overwritten by
     * {@code updateActiveCrop} to always point at the nearest unharvested
     * crop. The original nav goal (gold-block marker) is stored here so the
     * multitask success condition can correctly check distance to the marker
     * rather than the last harvested crop. (Fix 4.1)
     */
    public double  navTargetX = 8.0;
    public double  navTargetZ = 0.0;

    // ------------------------------------------------------------------
    // Curriculum / experiment config (set on each reset)
    // ------------------------------------------------------------------
    public boolean sparseReward           = false;
    public double  curriculumMinDist      = -1.0;
    public double  curriculumMaxDist      = -1.0;
    public int     curriculumNumObstacles = -1;

    // ------------------------------------------------------------------
    // Agent references
    // ------------------------------------------------------------------
    public UUID     agentUuid      = null;
    public BlockPos markerPos      = null;

    // ------------------------------------------------------------------
    // Farming state (multi-crop)
    // ------------------------------------------------------------------
    /** All soil positions for this episode (one per crop plot). */
    public List<BlockPos> farmingSoilPositions = new ArrayList<>();
    /** All crop positions (above each soil). */
    public List<BlockPos> farmingCropPositions = new ArrayList<>();
    /** Which crops have been successfully harvested this episode. */
    public List<Boolean>  cropHarvested        = new ArrayList<>();
    /** Total crops to harvest this episode (1-10). */
    public int            totalCrops           = 1;
    /** How many crops harvested so far. */
    public int            cropsHarvested       = 0;
    /** Primary navigation target for farming (nearest unharvested crop). */
    public int            activeCropIndex      = 0;
    /** Farming sub-phase: 0=navigate-to-crop, 1=harvest */
    public int            farmingPhase         = 0;
    /** True if we are using full farming cycle (till+plant+grow+harvest). */
    public boolean        fullFarmingCycle     = false;
    /** Crop growth stage (0=seeds planted, 7=mature). -1 = pre-planted. */
    public List<Integer>  cropGrowthStages     = new ArrayList<>();

    // Obstacle blocks placed for navigation
    public List<BlockPos> obstaclePositions = new ArrayList<>();

    // ------------------------------------------------------------------
    // Per-step action flags
    // ------------------------------------------------------------------
    public boolean lastInteractValid    = false;
    public boolean lastJumpedObstacle   = false;
    public boolean lastAttackValid      = false;
    public boolean lastEatValid         = false;
    public boolean lastSprinting        = false;

    // ------------------------------------------------------------------
    // Health & food (mirrored from entity each step)
    // ------------------------------------------------------------------
    public float   health      = 20.0f;
    public float   maxHealth   = 20.0f;
    public int     foodLevel   = 20;    // 0-20
    public float   saturation  = 5.0f;
    public boolean isDead      = false;

    // ------------------------------------------------------------------
    // Inventory tracking (slot indices)
    // ------------------------------------------------------------------
    public static final int SLOT_SWORD  = 0;
    public static final int SLOT_FOOD   = 1;
    public static final int SLOT_SEEDS  = 2;
    public static final int SLOT_HOE    = 3;
    public static final int SLOT_BONES  = 4;   // bonemeal
    public int     activeSlot  = 0;   // currently held item slot

    // ------------------------------------------------------------------
    // Combat state
    // ------------------------------------------------------------------
    /** UUIDs of hostile mobs spawned for this episode. */
    public List<UUID> hostileMobUuids = new ArrayList<>();
    /** How many mobs were killed this episode. */
    public int        mobsKilled      = 0;
    /** How many times the agent was hit. */
    public int        timesHit        = 0;
    /**
     * Episode step at which the last successful (full-damage) attack landed.
     * Initialised to a large negative value so the cooldown is always
     * expired at the start of a new episode. (FIX 3.2)
     */
    public int        lastAttackStep  = -100;
    /**
     * Distance to the nearest alive hostile mob, updated every step.
     * Used by RewardCalculator to provide a proximity incentive in combat.
     * Set to MAX_VALUE when no mobs are present. (Fix 4.3)
     */
    public double     nearestMobDist  = Double.MAX_VALUE;

    // ------------------------------------------------------------------
    // Pitch (vertical look) for human-like movement
    // ------------------------------------------------------------------
    public float targetPitch = 0.0f;   // smoothly blended toward this
    public float currentPitch = 0.0f;

    // ------------------------------------------------------------------
    // Walk / run
    // ------------------------------------------------------------------
    public boolean isSprinting = false;

    // ------------------------------------------------------------------
    // Night mode flag
    // ------------------------------------------------------------------
    public boolean nightMode = true;   // always night so mobs don't burn

    // ------------------------------------------------------------------
    // Task helpers
    // ------------------------------------------------------------------

    public void setTask(String task) {
        switch (task.toLowerCase()) {
            case "farming" -> {
                this.taskName = "farming";
                this.taskId   = 1.0;
                this.maxSteps = 300;
            }
            case "combat" -> {
                this.taskName = "combat";
                this.taskId   = 2.0;
                // FIX 3.2: Increased from 200 → 400 so the agent has enough
                // time to navigate to mobs (6–10 blocks away) and kill all 3
                // with the 12-step attack cooldown.  200 steps was too tight.
                this.maxSteps = 400;
            }
            case "multitask" -> {
                this.taskName = "multitask";
                this.taskId   = 3.0;
                this.maxSteps = 400;
            }
            default -> {
                this.taskName = "navigation";
                this.taskId   = 0.0;
                this.maxSteps = 150;
            }
        }
    }

    public void reset(double initialDistance) {
        this.episodeStep        = 0;
        this.prevDistance       = initialDistance;
        this.stuckSteps         = 0;
        this.invalidActionCount = 0;
        this.done               = false;
        this.success            = false;
        this.lastInteractValid  = false;
        this.lastJumpedObstacle = false;
        this.lastAttackValid    = false;
        this.lastEatValid       = false;
        this.lastSprinting      = false;
        this.cropsHarvested     = 0;
        this.activeCropIndex    = 0;
        this.farmingPhase       = 0;
        this.mobsKilled         = 0;
        this.timesHit           = 0;
        this.isDead             = false;
        this.lastAttackStep     = -100;
        this.nearestMobDist     = Double.MAX_VALUE;
        // navTargetX/Z are set by TaskSetup on each reset and don't need clearing.
        this.isSprinting        = false;
        this.targetPitch        = 0.0f;
        this.currentPitch       = 0.0f;
        this.activeSlot         = 0;
        this.health             = 20.0f;
        this.foodLevel          = 20;
        this.saturation         = 5.0f;
    }

    /** Returns the BlockPos of the currently active (unharvested) crop, or null. */
    public BlockPos getActiveCropPos() {
        if (farmingCropPositions.isEmpty()) return null;
        for (int i = 0; i < farmingCropPositions.size(); i++) {
            if (i < cropHarvested.size() && !cropHarvested.get(i)) {
                return farmingCropPositions.get(i);
            }
        }
        return null;
    }

    /** Returns the index of the nearest unharvested crop to (x, z). */
    public int nearestUnharvestedCropIndex(double x, double z) {
        int best = -1;
        double bestDist = Double.MAX_VALUE;
        for (int i = 0; i < farmingCropPositions.size(); i++) {
            if (i < cropHarvested.size() && cropHarvested.get(i)) continue;
            BlockPos p = farmingCropPositions.get(i);
            double d = Math.sqrt(Math.pow(p.getX() - x, 2) + Math.pow(p.getZ() - z, 2));
            if (d < bestDist) { bestDist = d; best = i; }
        }
        return best;
    }

    /** Update active crop to nearest unharvested. */
    public void updateActiveCrop(double agentX, double agentZ) {
        int idx = nearestUnharvestedCropIndex(agentX, agentZ);
        if (idx >= 0) {
            activeCropIndex = idx;
            BlockPos p = farmingCropPositions.get(idx);
            targetX = p.getX() + 0.5;
            targetY = p.getY();
            targetZ = p.getZ() + 0.5;
        }
    }

    public boolean allCropsHarvested() {
        return cropsHarvested >= totalCrops;
    }
}