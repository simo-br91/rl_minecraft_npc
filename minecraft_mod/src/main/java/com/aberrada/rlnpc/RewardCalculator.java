package com.aberrada.rlnpc;

/**
 * Pure reward calculation — extracted from EnvironmentManager to reduce
 * the God-class problem.
 *
 * Reward design:
 *   NAVIGATION: progress shaping + success bonus + stuck/invalid penalties
 *   FARMING:    navigation signals + crop bonuses + harvest bonuses
 *   COMBAT:     kill bonus + survival bonus + damage penalties
 *   MULTITASK:  weighted combination based on current sub-task
 */
public class RewardCalculator {

    // Sparse mode
    private static final double SPARSE_SUCCESS   = 10.0;
    private static final double SPARSE_STEP      = -0.01;
    private static final double SPARSE_INVALID   = -0.10;
    private static final double SPARSE_STUCK     = -0.25;

    // Shaped: navigation
    private static final double PROGRESS_COEFF   = 0.15;
    private static final double TIME_PENALTY     = -0.015;
    private static final double JUMP_BONUS       = 0.30;   // reduced from 0.5 to be proportional
    private static final double NAV_SUCCESS      = 10.0;
    private static final double INVALID_PENALTY  = -0.10;
    private static final double STUCK_PENALTY    = -0.25;
    private static final int    STUCK_LIMIT      = 8;
    private static final double NAV_DRIFT_THRESHOLD  = 3.0;   // blocks
    private static final double NAV_DRIFT_STEP        = 0.05;  // min movement to count as drift
    private static final double NAV_DRIFT_PENALTY     = -0.10;

    // Shaped: farming
    private static final double NEAR_CROP_BONUS  = 0.05;
    private static final double CROP_FRONT_BONUS = 0.10;
    private static final double PRE_HARVEST_BONUS = 2.0;
    private static final double HARVEST_BONUS    = 4.0;
    private static final double ALL_HARVEST_BONUS = 10.0;
    private static final double INTERACT_MISS    = -0.05;
    private static final double BONEMEAL_BONUS   = 0.5;

    // Shaped: combat
    private static final double KILL_BONUS       = 8.0;
    private static final double ATTACKED_PENALTY = -1.0;
    private static final double ALL_KILLED_BONUS = 10.0;
    private static final double DEATH_PENALTY    = -5.0;
    private static final double SPRINT_FOOD_COST = -0.005;

    public static double compute(
            RLNpcEntity agent,
            EpisodeState state,
            int action,
            double beforeDistance,
            double currentDistance,
            boolean validAction,
            boolean success,
            boolean cropInFront) {

        if (state.sparseReward) {
            return computeSparse(state, validAction, success);
        }

        return switch (state.taskName) {
            case "farming"   -> computeFarming(agent, state, action, beforeDistance,
                                               currentDistance, validAction, success, cropInFront);
            case "combat"    -> computeCombat(state, validAction, success);
            case "multitask" -> computeMultitask(agent, state, action, beforeDistance,
                                                  currentDistance, validAction, success, cropInFront);
            default          -> computeNavigation(state, beforeDistance, currentDistance,
                                                   validAction, success);
        };
    }

    // ------------------------------------------------------------------
    // Sparse
    // ------------------------------------------------------------------

    private static double computeSparse(EpisodeState state, boolean validAction, boolean success) {
        if (success)           return SPARSE_SUCCESS;
        if (!validAction)      return SPARSE_INVALID;
        if (state.stuckSteps >= STUCK_LIMIT) return SPARSE_STUCK;
        return SPARSE_STEP;
    }

    // ------------------------------------------------------------------
    // Navigation
    // ------------------------------------------------------------------

    private static double computeNavigation(EpisodeState state,
            double before, double after, boolean valid, boolean success) {
        double r = 0.0;
        r += PROGRESS_COEFF * (before - after);
        r += TIME_PENALTY;
        // Drift penalty: penalise moving away when already close to target
        if (before < NAV_DRIFT_THRESHOLD && after > before + NAV_DRIFT_STEP) {
            r += NAV_DRIFT_PENALTY;
        }
        if (state.lastJumpedObstacle) r += JUMP_BONUS;
        if (success)                  r += NAV_SUCCESS;
        if (!valid)                   r += INVALID_PENALTY;
        if (state.stuckSteps >= STUCK_LIMIT) r += STUCK_PENALTY;
        if (state.lastSprinting)      r += SPRINT_FOOD_COST;
        return r;
    }

    // ------------------------------------------------------------------
    // Farming
    // ------------------------------------------------------------------

    private static double computeFarming(RLNpcEntity agent, EpisodeState state, int action,
            double before, double after, boolean valid, boolean success, boolean cropInFront) {
        double r = 0.0;
        r += PROGRESS_COEFF * (before - after);
        r += TIME_PENALTY;

        boolean nearCrop = after <= 1.10;
        if (nearCrop)    r += NEAR_CROP_BONUS;
        if (cropInFront) r += CROP_FRONT_BONUS;

        // Drift penalty: agent moves away after being very close
        if (before <= 1.15 && after > before + 1e-6) r -= 0.10;

        // Interact action
        if (action == 3) {
            if (cropInFront) r += PRE_HARVEST_BONUS;
            else             r += INTERACT_MISS;
        }
        if (state.lastInteractValid) r += HARVEST_BONUS;

        // Bonemeal applied (action 3 on non-mature crop with bonemeal)
        // future: add bonemeal logic

        // All crops done
        if (success) r += ALL_HARVEST_BONUS;

        if (state.lastJumpedObstacle) r += JUMP_BONUS;
        if (!valid)                   r += INVALID_PENALTY;
        if (state.stuckSteps >= STUCK_LIMIT) r += STUCK_PENALTY;
        if (state.lastSprinting)      r += SPRINT_FOOD_COST;
        return r;
    }

    // ------------------------------------------------------------------
    // Combat
    // ------------------------------------------------------------------

    private static double computeCombat(EpisodeState state, boolean valid, boolean success) {
        double r = 0.0;
        r += TIME_PENALTY;
        if (state.lastAttackValid) r += 1.0;   // swing connected
        if (state.mobsKilled > 0)  r += KILL_BONUS * state.mobsKilled;
        if (state.timesHit > 0)    r += ATTACKED_PENALTY * state.timesHit;
        if (state.isDead)          r += DEATH_PENALTY;
        if (success)               r += ALL_KILLED_BONUS;
        if (!valid)                r += INVALID_PENALTY;
        if (state.stuckSteps >= STUCK_LIMIT) r += STUCK_PENALTY;
        return r;
    }

    // ------------------------------------------------------------------
    // Multitask: agent manages all sub-tasks simultaneously
    // ------------------------------------------------------------------

    private static double computeMultitask(RLNpcEntity agent, EpisodeState state, int action,
            double before, double after, boolean valid, boolean success, boolean cropInFront) {
        double r = 0.0;
        r += TIME_PENALTY;

        // Navigation component (toward current target)
        r += 0.08 * (before - after);

        // Farming component
        if (cropInFront)             r += CROP_FRONT_BONUS * 0.5;
        if (action == 3 && cropInFront) r += PRE_HARVEST_BONUS * 0.5;
        if (state.lastInteractValid) r += HARVEST_BONUS * 0.5;

        // Combat component
        if (state.lastAttackValid)   r += 2.0;
        if (state.mobsKilled > 0)    r += KILL_BONUS * state.mobsKilled;
        if (state.timesHit > 0)      r += ATTACKED_PENALTY * state.timesHit;
        if (state.isDead)            r += DEATH_PENALTY;

        // Food management reward
        if (state.lastEatValid && state.foodLevel < 10) r += 1.0;

        if (success)               r += ALL_HARVEST_BONUS;
        if (state.lastJumpedObstacle) r += JUMP_BONUS;
        if (!valid)                r += INVALID_PENALTY;
        if (state.stuckSteps >= STUCK_LIMIT) r += STUCK_PENALTY;
        if (state.lastSprinting)   r += SPRINT_FOOD_COST;
        return r;
    }
}