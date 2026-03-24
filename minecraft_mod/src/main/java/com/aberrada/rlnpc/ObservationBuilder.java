package com.aberrada.rlnpc;

import net.minecraft.core.BlockPos;
import net.minecraft.world.entity.monster.Monster;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.block.CropBlock;
import net.minecraft.world.level.block.FarmBlock;
import net.minecraft.world.level.block.GrassBlock;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.Vec3;

import java.util.List;
import java.util.Locale;

/**
 * Builds a 28-element observation vector for the RL NPC.
 *
 * Index  Feature
 * -----  -------
 *   0    dx_norm           (dx / MAX_DIST, clamped ±1)
 *   1    dz_norm           (dz / MAX_DIST, clamped ±1)
 *   2    distance_norm     (distance / MAX_DIST)
 *   3    angle_to_target   (sin of angle error: facing vs target direction)
 *   4    yaw_norm          (yaw / 180, in [-1, 1])
 *   5    pitch_norm        (pitch / 90, in [-1, 1])
 *   6    blocked_front     (1 if solid block ahead at feet level)
 *   7    on_ground         (1 if solid block below)
 *   8    stuck_norm        (stuck_steps / 15, clamped [0,1])  ← was /10
 *   9    task_id_norm      (task_id / 3.0)
 *  10    crop_in_front     (1 if mature crop directly ahead)
 *  11    near_crop         (1 if within 1.5 blocks of any unharvested crop)
 *  12    obstacle_1block   (1 if jumpable 1-block wall ahead)
 *  13    health_norm       (health / 20)
 *  14    food_norm         (foodLevel / 20)
 *  15    mob_nearby        (1 if hostile mob within 8 blocks)
 *  16    mob_dist_norm     (distance to nearest mob / 8, clamped [0,1])
 *  17    mob_angle         (sin of angle to nearest mob)
 *  18    active_slot_norm  (activeSlot / 4)
 *  19    holding_sword     (1 if active slot is sword)
 *  20    holding_food      (1 if active slot is food)
 *  21    crops_remaining_norm  (remaining_crops / totalCrops)
 *  22    block_N           (1 if solid block 1.5 ahead at N)
 *  23    block_E           (1 if solid block 1.5 ahead at E)
 *  24    block_S           (1 if solid block 1.5 ahead at S)
 *  25    block_W           (1 if solid block 1.5 ahead at W)
 *  26    farmland_ahead    (1 if untilled soil directly ahead)
 *  27    has_seed          (1 if agent has seeds — always true in simulation)
 *  28    height_above_gnd  (agent Y − surface Y, normalised by 5 blocks, clamped [0,1])
 *
 * Changes vs previous version
 * ----------------------------
 * stuck_norm now divides by 15 instead of 10.  The old divisor caused
 * the feature to saturate at 1.0 when stuckSteps >= 10, making the policy
 * blind to prolonged stuckness.  With /15 the signal remains informative
 * up to 15 consecutive stuck steps before clamping.
 */
public class ObservationBuilder {

    public static final int OBS_DIM  = 29;
    private static final double MAX_DIST    = 30.0;
    private static final double MOB_RANGE   = 8.0;
    // FIX: raised from 10.0 → 15.0 to avoid premature saturation
    private static final double STUCK_DIVISOR = 15.0;

    public static double[] build(RLNpcEntity agent, EpisodeState state) {
        double dx   = state.targetX - agent.getX();
        double dz   = state.targetZ - agent.getZ();
        double dist = Math.sqrt(dx * dx + dz * dz);

        double dxNorm   = Math.max(-1.0, Math.min(1.0, dx / MAX_DIST));
        double dzNorm   = Math.max(-1.0, Math.min(1.0, dz / MAX_DIST));
        double distNorm = Math.min(1.0, dist / MAX_DIST);

        double angleToTarget = computeAngleToTarget(agent, dx, dz);

        double yawNorm   = normalizeYaw(agent.getYRot()) / 180.0;
        double pitchNorm = Math.max(-1.0, Math.min(1.0, agent.getXRot() / 90.0));

        double blockedFront = isBlockedFront(agent) ? 1.0 : 0.0;
        double onGround     = isOnGround(agent) ? 1.0 : 0.0;

        // FIX: divide by STUCK_DIVISOR (15) instead of 10
        double stuckNorm = Math.min(state.stuckSteps / STUCK_DIVISOR, 1.0);

        double taskIdNorm = state.taskId / 3.0;

        double cropInFront = isMatureCropInFront(agent) ? 1.0 : 0.0;
        double nearCrop    = isNearUnharvestedCrop(agent, state) ? 1.0 : 0.0;
        double obstacle1   = is1BlockObstacleAhead(agent) ? 1.0 : 0.0;

        double healthNorm = Math.max(0.0, Math.min(1.0, state.health / 20.0));
        double foodNorm   = Math.max(0.0, Math.min(1.0, state.foodLevel / 20.0));

        double[] mobInfo   = getMobInfo(agent);
        double mobNearby   = mobInfo[0];
        double mobDistNorm = mobInfo[1];
        double mobAngle    = mobInfo[2];

        double activeSlotNorm = state.activeSlot / 4.0;
        double holdingSword   = (state.activeSlot == EpisodeState.SLOT_SWORD) ? 1.0 : 0.0;
        double holdingFood    = (state.activeSlot == EpisodeState.SLOT_FOOD)  ? 1.0 : 0.0;

        double cropsRemainingNorm = state.totalCrops > 0
                ? (double)(state.totalCrops - state.cropsHarvested) / state.totalCrops
                : 0.0;

        double[] cardinal      = getCardinalBlocks(agent);
        double farmlandAhead   = isFarmlandAhead(agent) ? 1.0 : 0.0;
        double hasSeed         = 1.0;  // agent always has seeds in simulation

        double groundY           = ActionExecutor.findSurfaceY(agent, agent.getX(), agent.getY(), agent.getZ());
        double heightAboveGround = Math.max(0.0, Math.min(1.0, (agent.getY() - groundY) / 5.0));

        return new double[] {
            dxNorm,             //  0
            dzNorm,             //  1
            distNorm,           //  2
            angleToTarget,      //  3
            yawNorm,            //  4
            pitchNorm,          //  5
            blockedFront,       //  6
            onGround,           //  7
            stuckNorm,          //  8  ← fixed divisor
            taskIdNorm,         //  9
            cropInFront,        // 10
            nearCrop,           // 11
            obstacle1,          // 12
            healthNorm,         // 13
            foodNorm,           // 14
            mobNearby,          // 15
            mobDistNorm,        // 16
            mobAngle,           // 17
            activeSlotNorm,     // 18
            holdingSword,       // 19
            holdingFood,        // 20
            cropsRemainingNorm, // 21
            cardinal[0],        // 22  N
            cardinal[1],        // 23  E
            cardinal[2],        // 24  S
            cardinal[3],        // 25  W
            farmlandAhead,      // 26
            hasSeed,            // 27
            heightAboveGround,  // 28
        };
    }

    // ------------------------------------------------------------------
    // Angle to target
    // ------------------------------------------------------------------

    private static double computeAngleToTarget(RLNpcEntity agent, double dx, double dz) {
        if (Math.abs(dx) < 1e-6 && Math.abs(dz) < 1e-6) return 0.0;
        Vec3 flat = ActionExecutor.getHorizontalLook(agent);
        if (flat == null) return 0.0;
        double targetLen = Math.sqrt(dx * dx + dz * dz);
        double tdx = dx / targetLen, tdz = dz / targetLen;
        return flat.x * tdz - flat.z * tdx;
    }

    // ------------------------------------------------------------------
    // Block checks
    // ------------------------------------------------------------------

    private static double normalizeYaw(float yaw) {
        double r = yaw;
        while (r <= -180.0) r += 360.0;
        while (r >   180.0) r -= 360.0;
        return r;
    }

    private static boolean isBlockedFront(RLNpcEntity agent) {
        BlockPos feet = ActionExecutor.frontBlockPos(agent);
        BlockPos head = feet.above();
        BlockState fs = agent.level().getBlockState(feet);
        BlockState hs = agent.level().getBlockState(head);
        return (!fs.isAir() && fs.blocksMotion()) || (!hs.isAir() && hs.blocksMotion());
    }

    public static boolean isOnGround(RLNpcEntity agent) {
        BlockPos below = BlockPos.containing(agent.getX(), agent.getY() - 0.1, agent.getZ());
        BlockState bs  = agent.level().getBlockState(below);
        return !bs.isAir() && bs.blocksMotion();
    }

    public static boolean is1BlockObstacleAhead(RLNpcEntity agent) {
        Vec3 flat = ActionExecutor.getHorizontalLook(agent);
        if (flat == null) return false;
        double tx = agent.getX() + flat.x * 0.9;
        double tz = agent.getZ() + flat.z * 0.9;
        double y  = agent.getY();
        boolean low  = ActionExecutor.isBlockedAt(agent, tx, y, tz);
        boolean high = ActionExecutor.isBlockedAt(agent, tx, y + 1.0, tz);
        return low && !high;
    }

    private static double[] getCardinalBlocks(RLNpcEntity agent) {
        double x = agent.getX(), y = agent.getY(), z = agent.getZ();
        // N(−z), E(+x), S(+z), W(−x)
        double[] dirs = {0, -1,  1, 0,  0, 1,  -1, 0};
        double[] result = new double[4];
        for (int i = 0; i < 4; i++) {
            double ddx = dirs[i * 2], ddz = dirs[i * 2 + 1];
            result[i] = ActionExecutor.isBlockedAt(agent,
                    x + ddx * 1.5, y, z + ddz * 1.5) ? 1.0 : 0.0;
        }
        return result;
    }

    private static boolean isFarmlandAhead(RLNpcEntity agent) {
        BlockPos front = ActionExecutor.frontBlockPos(agent);
        BlockPos below = front.below();
        BlockState b = agent.level().getBlockState(below);
        return b.getBlock() instanceof GrassBlock
            || b.getBlock().getClass().getSimpleName().contains("Dirt");
    }

    // ------------------------------------------------------------------
    // Crop helpers
    // ------------------------------------------------------------------

    public static BlockPos frontCropPos(RLNpcEntity agent) {
        Vec3 flat = ActionExecutor.getHorizontalLook(agent);
        if (flat == null) flat = new Vec3(0, 0, 1);
        // Use the same Y offset as frontBlockPos for consistency
        Vec3 origin = agent.position().add(0.0, 0.2, 0.0);
        Vec3 front  = origin.add(flat.scale(0.9));
        return BlockPos.containing(front.x, origin.y, front.z);
    }

    public static boolean isMatureCropInFront(RLNpcEntity agent) {
        BlockPos pos = frontCropPos(agent);
        BlockState b = agent.level().getBlockState(pos);
        return b.getBlock() instanceof CropBlock crop && crop.isMaxAge(b);
    }

    private static boolean isNearUnharvestedCrop(RLNpcEntity agent, EpisodeState state) {
        for (int i = 0; i < state.farmingCropPositions.size(); i++) {
            if (i < state.cropHarvested.size() && state.cropHarvested.get(i)) continue;
            BlockPos p = state.farmingCropPositions.get(i);
            double dist = Math.sqrt(Math.pow(p.getX() - agent.getX(), 2)
                                  + Math.pow(p.getZ() - agent.getZ(), 2));
            if (dist <= 1.5) return true;
        }
        return false;
    }

    // ------------------------------------------------------------------
    // Mob sensing
    // ------------------------------------------------------------------

    private static double[] getMobInfo(RLNpcEntity agent) {
        AABB box = agent.getBoundingBox().inflate(MOB_RANGE);
        List<Monster> mobs = agent.level().getEntitiesOfClass(
                Monster.class, box, m -> m.isAlive() && !m.isSpectator());
        if (mobs.isEmpty()) return new double[]{0.0, 1.0, 0.0};

        Monster nearest = null;
        double bestDist = Double.MAX_VALUE;
        for (Monster m : mobs) {
            double d = m.distanceTo(agent);
            if (d < bestDist) { bestDist = d; nearest = m; }
        }
        if (nearest == null) return new double[]{0.0, 1.0, 0.0};

        double distNorm = Math.min(1.0, bestDist / MOB_RANGE);
        double mdx = nearest.getX() - agent.getX();
        double mdz = nearest.getZ() - agent.getZ();
        double mLen = Math.sqrt(mdx * mdx + mdz * mdz);
        double sinAngle = 0.0;
        if (mLen > 1e-6) {
            Vec3 flat = ActionExecutor.getHorizontalLook(agent);
            if (flat != null) {
                sinAngle = flat.x * (mdz / mLen) - flat.z * (mdx / mLen);
            }
        }
        return new double[]{1.0, distNorm, sinAngle};
    }

    // ------------------------------------------------------------------
    // Serialization
    // ------------------------------------------------------------------

    public static String obsToJson(double[] obs) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < obs.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(String.format(Locale.US, "%.6f", obs[i]));
        }
        sb.append("]");
        return sb.toString();
    }
}