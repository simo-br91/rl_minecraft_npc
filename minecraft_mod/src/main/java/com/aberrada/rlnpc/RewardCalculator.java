package com.aberrada.rlnpc;

/**
 * Pure reward calculation.
 *
 * Fixes vs previous version:
 *  - Bug 5.2 (stuck-near-target): a drift penalty now fires when the agent
 *    is within NEAR_TARGET_THRESHOLD blocks AND moves further away, preventing
 *    indefinite small rewards without reaching the goal.
 *  - Bug 5.3 (jump bonus disproportionate): reduced from 0.30 → 0.20, which
 *    is now closer to a single step's progress reward on a 7-block episode.
 *  - Issue 5.4 (no_op behaviour): no_op (action 4) always returns valid=true
 *    so it never incurs the invalid-action penalty. This is intentional —
 *    the time penalty (−0.015/step) is sufficient deterrent against doing
 *    nothing, and penalising no_op as "invalid" would conflate two different
 *    semantics. Documented here for clarity.
 *  - Bug 3.3 (inverted nav drift condition): NAV_DRIFT_PENALTY was guarded
 *    by `before < NAV_DRIFT_THRESHOLD` which fired when the agent was CLOSE
 *    to the target, not far. Corrected to `before > NAV_DRIFT_THRESHOLD`.
 *  - Bug 3.4 (combat stuck penalty): the stuck counter increments when the
 *    nav-target distance doesn't change, which is normal in combat (fighting
 *    in place). The stuck penalty is now suppressed in combat mode to avoid
 *    penalising the agent for a valid in-place fighting strategy.
 */
public class RewardCalculator {

    // Sparse mode
    private static final double SPARSE_SUCCESS    = 10.0;
    private static final double SPARSE_STEP       = -0.01;
    private static final double SPARSE_INVALID    = -0.10;
    private static final double SPARSE_STUCK      = -0.25;

    // Shaped: navigation
    private static final double PROGRESS_COEFF    = 0.15;
    private static final double TIME_PENALTY      = -0.015;
    /**
     * FIX 5.3: Jump bonus reduced from 0.30 → 0.20.
     * A 7-block navigation episode yields roughly 0.15 × 7 = 1.05 total
     * progress reward.  At 0.30 a single jump was worth ~28% of the whole
     * episode, biasing the agent to seek obstacles.  0.20 (≈19%) is more
     * proportional while still providing a meaningful incentive.
     */
    private static final double JUMP_BONUS        = 0.20;
    private static final double NAV_SUCCESS       = 10.0;
    private static final double INVALID_PENALTY   = -0.10;
    private static final double STUCK_PENALTY     = -0.25;
    private static final int    STUCK_LIMIT       = 8;
    private static final double NAV_DRIFT_THRESHOLD = 3.0;
    private static final double NAV_DRIFT_STEP    = 0.05;
    private static final double NAV_DRIFT_PENALTY = -0.10;

    /**
     * FIX 5.2: Near-target drift penalty.
     * When the agent is within NEAR_TARGET_THRESHOLD blocks and moves further
     * away, apply an additional penalty to prevent indefinite small rewards
     * from oscillating near (but never reaching) the target.
     */
    private static final double NEAR_TARGET_THRESHOLD = 1.5;
    private static final double NEAR_TARGET_DRIFT_PENALTY = -0.20;

    // Shaped: farming
    private static final double NEAR_CROP_BONUS   = 0.05;
    private static final double CROP_FRONT_BONUS  = 0.10;
    private static final double PRE_HARVEST_BONUS = 2.0;
    private static final double HARVEST_BONUS     = 4.0;
    private static final double ALL_HARVEST_BONUS = 10.0;
    private static final double INTERACT_MISS     = -0.05;

    // Shaped: combat
    private static final double KILL_BONUS        = 8.0;
    private static final double ATTACKED_PENALTY  = -1.0;
    private static final double ALL_KILLED_BONUS  = 10.0;
    private static final double DEATH_PENALTY     = -5.0;
    private static final double SPRINT_FOOD_COST  = -0.005;
    /** Proximity incentive: reward for staying within melee range of a mob. (Fix 4.3) */
    private static final double MOB_PROXIMITY_BONUS = 0.05;
    private static final double MOB_PROXIMITY_RANGE = ActionExecutor.ATTACK_RANGE + 0.5;

    public static double compute(
            RLNpcEntity  agent,
            EpisodeState state,
            int          action,
            double       beforeDistance,
            double       currentDistance,
            boolean      validAction,
            boolean      success,
            boolean      cropInFront) {

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
        if (success)                           return SPARSE_SUCCESS;
        if (!validAction)                      return SPARSE_INVALID;
        if (state.stuckSteps >= STUCK_LIMIT)   return SPARSE_STUCK;
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

        // Drift penalty when far from target (> 3 blocks) and moving away.
        // FIX 3.3: was `before < NAV_DRIFT_THRESHOLD` (wrong — penalised near
        // target only). Corrected to `before > NAV_DRIFT_THRESHOLD`.
        if (before > NAV_DRIFT_THRESHOLD && after > before + NAV_DRIFT_STEP) {
            r += NAV_DRIFT_PENALTY;
        }

        // FIX 5.2: near-target drift penalty — prevents indefinite oscillation
        if (before <= NEAR_TARGET_THRESHOLD && after > before) {
            r += NEAR_TARGET_DRIFT_PENALTY;
        }

        if (state.lastJumpedObstacle) r += JUMP_BONUS;  // FIX 5.3: 0.20 not 0.30
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
        // FIX 4.4a: NEAR_CROP_BONUS only fires when the agent is actively
        // closing distance (after < before - 0.005).  Previously it fired
        // every step within 1.10 blocks regardless of movement direction,
        // creating a local optimum of orbiting the crop at constant range
        // without harvesting.
        if (nearCrop && after < before - 0.005) r += NEAR_CROP_BONUS;
        if (cropInFront) r += CROP_FRONT_BONUS;

        // FIX 5.2: drift penalty when very close to crop
        if (before <= 1.15 && after > before) r += NEAR_TARGET_DRIFT_PENALTY;

        if (action == 3) {
            // FIX 4.4b: PRE_HARVEST_BONUS and HARVEST_BONUS both fired on the
            // same successful interact step (total +6.0), because cropInFront
            // and lastInteractValid were always both true when a harvest succeeded.
            // The per-step CROP_FRONT_BONUS already incentivises facing the crop;
            // PRE_HARVEST is now suppressed on the harvest step to prevent
            // double-counting.  On a genuine "interact while facing ripe crop but
            // harvest failed" edge case, PRE_HARVEST still fires as intended.
            if (cropInFront && !state.lastInteractValid) r += PRE_HARVEST_BONUS;
            else if (!cropInFront)                       r += INTERACT_MISS;
        }
        if (state.lastInteractValid) r += HARVEST_BONUS;
        if (success)                 r += ALL_HARVEST_BONUS;
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
        if (state.lastAttackValid) r += 1.0;
        if (state.mobsKilled > 0)  r += KILL_BONUS * state.mobsKilled;
        if (state.timesHit > 0)    r += ATTACKED_PENALTY * state.timesHit;
        if (state.isDead)          r += DEATH_PENALTY;
        if (success)               r += ALL_KILLED_BONUS;
        if (!valid)                r += INVALID_PENALTY;
        // Fix 4.3: proximity bonus — small per-step reward for staying in
        // melee range.  Without this there is no shaping toward mobs; the
        // agent only earns rewards when it attacks (requires cooldown) or
        // kills (delayed).  The bonus counteracts the incentive to kite away
        // from mobs to avoid the -1.0 damage penalty.
        if (state.nearestMobDist <= MOB_PROXIMITY_RANGE) r += MOB_PROXIMITY_BONUS;
        // FIX 3.4: Stuck penalty removed for combat.
        // The stuck counter is based on nav-target distance, which barely
        // changes when the agent is fighting in place — a completely valid
        // and optimal combat strategy (mobs come to the agent).  Applying
        // the stuck penalty here incorrectly penalises standing and fighting.
        return r;
    }

    // ------------------------------------------------------------------
    // Multitask
    // ------------------------------------------------------------------

    private static double computeMultitask(RLNpcEntity agent, EpisodeState state, int action,
            double before, double after, boolean valid, boolean success, boolean cropInFront) {
        double r = 0.0;
        r += TIME_PENALTY;
        r += 0.08 * (before - after);

        // FIX 5.2: near-target drift in multitask too
        if (before <= NEAR_TARGET_THRESHOLD && after > before) {
            r += NEAR_TARGET_DRIFT_PENALTY * 0.5;
        }

        if (cropInFront)              r += CROP_FRONT_BONUS * 0.5;
        if (action == 3 && cropInFront) r += PRE_HARVEST_BONUS * 0.5;
        if (state.lastInteractValid)  r += HARVEST_BONUS * 0.5;
        if (state.lastAttackValid)    r += 2.0;
        if (state.mobsKilled > 0)     r += KILL_BONUS * state.mobsKilled;
        if (state.timesHit > 0)       r += ATTACKED_PENALTY * state.timesHit;
        if (state.isDead)             r += DEATH_PENALTY;
        if (state.lastEatValid && state.foodLevel < 10) r += 1.0;
        if (success)                  r += ALL_HARVEST_BONUS;
        if (state.lastJumpedObstacle) r += JUMP_BONUS;
        if (!valid)                   r += INVALID_PENALTY;
        if (state.stuckSteps >= STUCK_LIMIT) r += STUCK_PENALTY;
        if (state.lastSprinting)      r += SPRINT_FOOD_COST;
        return r;
    }
}
