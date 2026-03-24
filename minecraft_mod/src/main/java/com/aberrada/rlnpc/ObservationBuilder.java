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
 * Builds the observation vector for the RL NPC.
 *
 * Layout (OBS_DIM = 45):
 *
 * Core navigation / position (0-8):
 *   0   dx_norm            dx / MAX_DIST, clamped ±1
 *   1   dz_norm            dz / MAX_DIST, clamped ±1
 *   2   distance_norm      Euclidean dist / MAX_DIST
 *   3   angle_to_target    sin of yaw-error to target
 *   4   yaw_norm           yaw / 180
 *   5   pitch_norm         pitch / 90
 *   6   blocked_front      solid block directly ahead (feet+head)
 *   7   on_ground          solid block below
 *   8   stuck_norm         stuckSteps / 15, clamped [0,1]
 *
 * Task / agent state (9-14):
 *   9   task_id_norm       taskId / 3.0
 *  10   health_norm        health / 20
 *  11   food_norm          foodLevel / 20
 *  12   height_above_gnd   (agentY - surfaceY) / 5, clamped [0,1]
 *  13   active_slot_norm   activeSlot / 4
 *  14   holding_sword      1 if slot == SWORD
 *
 * Farming (15-18):
 *  15   crop_in_front      mature wheat directly ahead
 *  16   near_crop          within 1.5 blocks of unharvested crop
 *  17   crops_remaining    (total - harvested) / total
 *  18   holding_food       1 if slot == FOOD
 *
 * Combat / mobs (19-21):
 *  19   mob_nearby         hostile mob within 8 blocks
 *  20   mob_dist_norm      dist to nearest mob / 8
 *  21   mob_angle          sin of angle to nearest mob
 *
 * Navigation obstacles (22-26):
 *  22   obstacle_1block    jumpable 1-block wall ahead
 *  23   block_N            solid 1.5 blocks N
 *  24   block_E            solid 1.5 blocks E
 *  25   block_S            solid 1.5 blocks S
 *  26   block_W            solid 1.5 blocks W
 *
 * Farming extras (27-28):
 *  27   farmland_ahead     untilled soil directly ahead
 *  28   has_seed           always 1 in simulation
 *
 * Voxel grid — 5×3×5 (XZY) local block scan, 75 cells (29-103 → 29-103)
 * Each cell: 0.0 = air/passable, 1.0 = solid.
 * Origin: agent feet position. X/Z scan ±2 blocks, Y scan 0..+2.
 * Indices 29 .. 28+75 = 29..103  →  OBS_DIM = 104
 *
 * Changes vs previous version
 * ----------------------------
 * - near_target binary (was index 9) replaced by mob_angle (was 17) reorganised;
 *   the redundant near_target flag (distance ≤ 1.10) has been removed since the
 *   policy already receives the exact distance_norm.
 * - frontCropPos now uses the same Y offset as frontBlockPos (0.2, not 0.5) so
 *   collision checks and crop checks sample from the same XZ plane. (Bug 2.11)
 * - Local 5×3×5 voxel grid appended (indices 29-103). (Req 1.1)
 * - build() now accepts a read-only ObsSnapshot instead of mutable EpisodeState
 *   so it can be called for logging/replay without a live episode. (Issue 7.6)
 *   A convenience overload build(agent, state) still exists for production use.
 */
public class ObservationBuilder {

    public static final int OBS_DIM = 104;

    // Feature constants
    private static final double MAX_DIST      = 30.0;
    private static final double MOB_RANGE     = 8.0;
    private static final double STUCK_DIVISOR = 15.0;

    // Voxel grid: X from -2..+2, Z from -2..+2, Y from 0..+2
    private static final int GRID_HALF  = 2;   // radius
    private static final int GRID_W     = 5;   // 2*HALF+1
    private static final int GRID_H     = 3;   // Y levels 0,1,2
    private static final int GRID_CELLS = GRID_W * GRID_W * GRID_H; // 75

    // ------------------------------------------------------------------
    // Public API — production path (uses live EpisodeState)
    // ------------------------------------------------------------------

    public static double[] build(RLNpcEntity agent, EpisodeState state) {
        return buildFromSnapshot(agent, ObsSnapshot.from(state));
    }

    // ------------------------------------------------------------------
    // Public API — replay/logging path (uses immutable snapshot)
    // ------------------------------------------------------------------

    public static double[] buildFromSnapshot(RLNpcEntity agent, ObsSnapshot snap) {
        double dx   = snap.targetX - agent.getX();
        double dz   = snap.targetZ - agent.getZ();
        double dist = Math.sqrt(dx * dx + dz * dz);

        double dxNorm   = clamp(dx / MAX_DIST, -1.0, 1.0);
        double dzNorm   = clamp(dz / MAX_DIST, -1.0, 1.0);
        double distNorm = Math.min(1.0, dist / MAX_DIST);

        double angleToTarget = computeAngleToTarget(agent, dx, dz);
        double yawNorm       = normalizeYaw(agent.getYRot()) / 180.0;
        double pitchNorm     = clamp(agent.getXRot() / 90.0, -1.0, 1.0);

        double blockedFront = isBlockedFront(agent) ? 1.0 : 0.0;
        double onGround     = isOnGround(agent)     ? 1.0 : 0.0;
        double stuckNorm    = Math.min(snap.stuckSteps / STUCK_DIVISOR, 1.0);

        double taskIdNorm = snap.taskId / 3.0;
        double healthNorm = clamp(snap.health / 20.0, 0.0, 1.0);
        double foodNorm   = clamp(snap.foodLevel / 20.0, 0.0, 1.0);

        double groundY           = ActionExecutor.findSurfaceY(agent, agent.getX(), agent.getY(), agent.getZ());
        double heightAboveGround = clamp((agent.getY() - groundY) / 5.0, 0.0, 1.0);
        double activeSlotNorm    = snap.activeSlot / 4.0;
        double holdingSword      = (snap.activeSlot == EpisodeState.SLOT_SWORD) ? 1.0 : 0.0;

        // Farming
        boolean cropFront = isMatureCropInFront(agent);
        double cropInFront = cropFront ? 1.0 : 0.0;
        double nearCrop    = isNearUnharvestedCrop(agent, snap) ? 1.0 : 0.0;
        double cropsRemainingNorm = snap.totalCrops > 0
                ? (double)(snap.totalCrops - snap.cropsHarvested) / snap.totalCrops
                : 0.0;
        double holdingFood = (snap.activeSlot == EpisodeState.SLOT_FOOD) ? 1.0 : 0.0;

        // Combat
        double[] mobInfo    = getMobInfo(agent);
        double mobNearby    = mobInfo[0];
        double mobDistNorm  = mobInfo[1];
        double mobAngle     = mobInfo[2];

        // Obstacle / navigation
        double obstacle1   = is1BlockObstacleAhead(agent) ? 1.0 : 0.0;
        double[] cardinal  = getCardinalBlocks(agent);
        double farmlandAhead = isFarmlandAhead(agent) ? 1.0 : 0.0;
        double hasSeed       = 1.0;

        // Voxel grid
        double[] voxel = buildVoxelGrid(agent);

        // Assemble
        double[] obs = new double[OBS_DIM];
        obs[ 0] = dxNorm;
        obs[ 1] = dzNorm;
        obs[ 2] = distNorm;
        obs[ 3] = angleToTarget;
        obs[ 4] = yawNorm;
        obs[ 5] = pitchNorm;
        obs[ 6] = blockedFront;
        obs[ 7] = onGround;
        obs[ 8] = stuckNorm;
        obs[ 9] = taskIdNorm;
        obs[10] = healthNorm;
        obs[11] = foodNorm;
        obs[12] = heightAboveGround;
        obs[13] = activeSlotNorm;
        obs[14] = holdingSword;
        obs[15] = cropInFront;
        obs[16] = nearCrop;
        obs[17] = cropsRemainingNorm;
        obs[18] = holdingFood;
        obs[19] = mobNearby;
        obs[20] = mobDistNorm;
        obs[21] = mobAngle;
        obs[22] = obstacle1;
        obs[23] = cardinal[0]; // N
        obs[24] = cardinal[1]; // E
        obs[25] = cardinal[2]; // S
        obs[26] = cardinal[3]; // W
        obs[27] = farmlandAhead;
        obs[28] = hasSeed;
        System.arraycopy(voxel, 0, obs, 29, GRID_CELLS);
        return obs;
    }

    // ------------------------------------------------------------------
    // Voxel grid (5×3×5 local scan) — Req 1.1
    // ------------------------------------------------------------------

    private static double[] buildVoxelGrid(RLNpcEntity agent) {
        double[] grid = new double[GRID_CELLS];
        int ox = (int) Math.floor(agent.getX());
        int oy = (int) Math.floor(agent.getY());
        int oz = (int) Math.floor(agent.getZ());

        int idx = 0;
        for (int dy = 0; dy < GRID_H; dy++) {
            for (int dz = -GRID_HALF; dz <= GRID_HALF; dz++) {
                for (int dx = -GRID_HALF; dx <= GRID_HALF; dx++) {
                    BlockPos pos = new BlockPos(ox + dx, oy + dy, oz + dz);
                    BlockState bs = agent.level().getBlockState(pos);
                    grid[idx++] = (!bs.isAir() && bs.blocksMotion()) ? 1.0 : 0.0;
                }
            }
        }
        return grid;
    }

    // ------------------------------------------------------------------
    // Angle to target
    // ------------------------------------------------------------------

    private static double computeAngleToTarget(RLNpcEntity agent, double dx, double dz) {
        if (Math.abs(dx) < 1e-6 && Math.abs(dz) < 1e-6) return 0.0;
        Vec3 flat = ActionExecutor.getHorizontalLook(agent);
        if (flat == null) return 0.0;
        double len = Math.sqrt(dx * dx + dz * dz);
        double tdx = dx / len, tdz = dz / len;
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
    // Crop helpers  — FIX 2.11: frontCropPos uses Y+0.2 (same as frontBlockPos)
    // ------------------------------------------------------------------

    /**
     * Returns the block position directly in front of the agent at feet level.
     * Uses the same Y offset (0.2) as {@link ActionExecutor#frontBlockPos} so
     * collision and crop checks sample from the same XZ plane.
     * (Previously used 0.5 which caused inconsistent results.)
     */
    public static BlockPos frontCropPos(RLNpcEntity agent) {
        Vec3 flat = ActionExecutor.getHorizontalLook(agent);
        if (flat == null) flat = new Vec3(0, 0, 1);
        // FIX 2.11: use 0.2 (same as frontBlockPos), not 0.5
        Vec3 origin = agent.position().add(0.0, 0.2, 0.0);
        Vec3 front  = origin.add(flat.scale(0.9));
        return BlockPos.containing(front.x, origin.y, front.z);
    }

    public static boolean isMatureCropInFront(RLNpcEntity agent) {
        BlockPos pos = frontCropPos(agent);
        BlockState b = agent.level().getBlockState(pos);
        return b.getBlock() instanceof CropBlock crop && crop.isMaxAge(b);
    }

    private static boolean isNearUnharvestedCrop(RLNpcEntity agent, ObsSnapshot snap) {
        for (int i = 0; i < snap.farmingCropPositions.size(); i++) {
            if (i < snap.cropHarvested.size() && snap.cropHarvested.get(i)) continue;
            BlockPos p = snap.farmingCropPositions.get(i);
            double d = Math.sqrt(Math.pow(p.getX() - agent.getX(), 2)
                               + Math.pow(p.getZ() - agent.getZ(), 2));
            if (d <= 1.5) return true;
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

        Monster nearest   = null;
        double  bestDist  = Double.MAX_VALUE;
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

    // ------------------------------------------------------------------
    // Utility
    // ------------------------------------------------------------------

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
