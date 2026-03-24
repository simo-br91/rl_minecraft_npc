package com.aberrada.rlnpc;

import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.CropBlock;
import net.minecraft.world.level.block.FarmBlock;
import net.minecraft.world.level.levelgen.Heightmap;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Encapsulates all task-specific world setup:
 * - Navigation obstacle placement
 * - Farming plot creation (multi-crop, full cycle or harvest-only)
 * - Combat mob spawning
 * - Artifact cleanup between episodes
 *
 * Extracted from EnvironmentManager to reduce the God-class.
 *
 * Y-resolution API (single source of truth):
 *   resolveGroundY   → the Y of the top solid block (i.e., the surface block itself)
 *   resolveStandY    → resolveGroundY + 1  (where the agent's feet go)
 *   getReferenceGroundY → integer version of resolveGroundY for block placement
 *
 * All three delegate to the same heightmap query.
 */
public class TaskSetup {

    // ------------------------------------------------------------------
    // Constants
    // ------------------------------------------------------------------

    private static final int    MAX_CROPS          = 10;
    private static final int    MAX_OBSTACLES       = 3;
    private static final double NAV_DEFAULT_MIN    = 7.0;
    private static final double NAV_DEFAULT_MAX    = 14.0;

    // Minimum safe clearance between an obstacle and spawn / target
    private static final double OBSTACLE_SPAWN_CLEARANCE  = 2.0;
    private static final double OBSTACLE_TARGET_CLEARANCE = 2.0;

    // ------------------------------------------------------------------
    // Navigation
    // ------------------------------------------------------------------

    public static void configureNavigation(ServerLevel level, EpisodeState state, Random rng) {
        double minD = state.curriculumMinDist  > 0 ? state.curriculumMinDist  : NAV_DEFAULT_MIN;
        double maxD = state.curriculumMaxDist  > 0 ? state.curriculumMaxDist  : NAV_DEFAULT_MAX;
        if (maxD < minD) maxD = minD + 1.0;

        double angle  = rng.nextDouble() * 2.0 * Math.PI;
        double dist   = minD + rng.nextDouble() * (maxD - minD);
        state.targetX = Math.round(Math.cos(angle) * dist);
        state.targetZ = Math.round(Math.sin(angle) * dist);
        state.targetY = resolveGroundY(level, state.targetX, state.targetZ) + 1.0;

        // Scale maxSteps linearly with target distance
        int base = (int)(dist * 15);
        state.maxSteps = Math.max(100, Math.min(300, base));
    }

    public static void placeNavigationObstacles(ServerLevel level, EpisodeState state,
                                                 double spawnX, double spawnZ, Random rng) {
        int requested = state.curriculumNumObstacles >= 0
                ? state.curriculumNumObstacles
                : 1 + rng.nextInt(2);

        // Cap at MAX_OBSTACLES and log a warning if clipped
        int count = requested;
        if (count > MAX_OBSTACLES) {
            RLNpcMod.LOGGER.warn(
                "placeNavigationObstacles: requested {} obstacles but max is {}; capping.",
                requested, MAX_OBSTACLES);
            count = MAX_OBSTACLES;
        }
        if (count == 0) return;

        double dx   = state.targetX - spawnX;
        double dz   = state.targetZ - spawnZ;
        double dist = Math.sqrt(dx * dx + dz * dz);
        if (dist < 4.0) return;

        // Place obstacles at evenly-spaced fractions along the path, with jitter
        double[] fracs = {0.30, 0.50, 0.65, 0.80};
        int placed = 0;
        for (int i = 0; i < fracs.length && placed < count; i++) {
            double frac = fracs[i] + (rng.nextDouble() - 0.5) * 0.06;
            int ox = (int) Math.round(spawnX + dx * frac);
            int oz = (int) Math.round(spawnZ + dz * frac);
            int gy = getReferenceGroundY(level, ox, oz);

            // Reject positions too close to the target OR to the spawn point
            double wallDistTarget = Math.sqrt(Math.pow(ox - state.targetX, 2)
                                            + Math.pow(oz - state.targetZ, 2));
            double wallDistSpawn  = Math.sqrt(Math.pow(ox - spawnX, 2)
                                            + Math.pow(oz - spawnZ, 2));
            if (wallDistTarget < OBSTACLE_TARGET_CLEARANCE
             || wallDistSpawn  < OBSTACLE_SPAWN_CLEARANCE) {
                continue;
            }

            BlockPos wallPos = new BlockPos(ox, gy + 1, oz);
            level.setBlockAndUpdate(wallPos, Blocks.STONE.defaultBlockState());
            state.obstaclePositions.add(wallPos);
            placed++;
        }
    }

    // ------------------------------------------------------------------
    // Farming — multi-crop with optional full cycle
    // ------------------------------------------------------------------

    /**
     * Configure farming task.
     *
     * @param numCrops   How many wheat plots to create (1–MAX_CROPS).
     * @param fullCycle  If true, place seeds (age=0) so the agent must
     *                   use bonemeal or wait.  If false, crops are pre-grown.
     */
    public static void configureFarming(ServerLevel level, EpisodeState state,
                                         int numCrops, boolean fullCycle, Random rng) {
        numCrops = Math.max(1, Math.min(MAX_CROPS, numCrops));
        state.totalCrops      = numCrops;
        state.fullFarmingCycle = fullCycle;
        state.farmingSoilPositions.clear();
        state.farmingCropPositions.clear();
        state.cropHarvested.clear();
        state.cropGrowthStages.clear();
        state.cropsHarvested  = 0;

        // Spread crops in a grid-ish pattern around a central point
        double farmAngle = rng.nextDouble() * 2.0 * Math.PI;
        double farmDist  = 5.0 + rng.nextDouble() * 10.0;
        int    centerX   = (int) Math.round(Math.cos(farmAngle) * farmDist);
        int    centerZ   = (int) Math.round(Math.sin(farmAngle) * farmDist);

        List<int[]> positions = generateCropPositions(centerX, centerZ, numCrops, rng);
        int groundY = getReferenceGroundY(level, centerX, centerZ);

        for (int[] pos : positions) {
            int cx = pos[0], cz = pos[1];
            BlockPos soilPos = new BlockPos(cx, groundY, cz);
            BlockPos cropPos = soilPos.above();

            level.setBlockAndUpdate(soilPos, Blocks.FARMLAND.defaultBlockState());

            CropBlock wheat = (CropBlock) Blocks.WHEAT;
            if (fullCycle) {
                // Plant seeds (age=0) — agent must use bonemeal or wait for growth
                level.setBlockAndUpdate(cropPos, wheat.getStateForAge(0));
                state.cropGrowthStages.add(0);
            } else {
                // Pre-grown wheat (age=max) — harvest immediately
                level.setBlockAndUpdate(cropPos, wheat.getStateForAge(wheat.getMaxAge()));
                state.cropGrowthStages.add(wheat.getMaxAge());
            }

            state.farmingSoilPositions.add(soilPos);
            state.farmingCropPositions.add(cropPos);
            state.cropHarvested.add(false);
        }

        // Set initial navigation target to the first crop
        if (!state.farmingCropPositions.isEmpty()) {
            BlockPos first = state.farmingCropPositions.get(0);
            state.targetX = first.getX() + 0.5;
            state.targetY = first.getY();
            state.targetZ = first.getZ() + 0.5;
            state.activeCropIndex = 0;
        }

        // Budget: base steps + extra per crop (more crops = more walking)
        state.maxSteps = 150 + numCrops * 80;
    }

    /** Generate distinct crop positions in a grid with random jitter. */
    private static List<int[]> generateCropPositions(int cx, int cz, int n, Random rng) {
        List<int[]> result = new ArrayList<>();
        int cols    = (int) Math.ceil(Math.sqrt(n));
        int spacing = 3;
        for (int i = 0; i < n; i++) {
            int row = i / cols;
            int col = i % cols;
            int x = cx + (col - cols / 2) * spacing + rng.nextInt(2) - 1;
            int z = cz + (row - cols / 2) * spacing + rng.nextInt(2) - 1;
            result.add(new int[]{x, z});
        }
        return result;
    }

    // ------------------------------------------------------------------
    // Artifact cleanup
    // ------------------------------------------------------------------

    public static void clearTaskArtifacts(ServerLevel level, EpisodeState state) {
        if (state.markerPos != null) {
            level.setBlockAndUpdate(state.markerPos, Blocks.AIR.defaultBlockState());
            state.markerPos = null;
        }
        for (BlockPos soilPos : state.farmingSoilPositions) {
            if (soilPos != null) {
                BlockPos cropPos = soilPos.above();
                level.setBlockAndUpdate(cropPos, Blocks.AIR.defaultBlockState());
                level.setBlockAndUpdate(soilPos, Blocks.GRASS_BLOCK.defaultBlockState());
            }
        }
        state.farmingSoilPositions.clear();
        state.farmingCropPositions.clear();
        state.cropHarvested.clear();
        state.cropGrowthStages.clear();

        for (BlockPos pos : state.obstaclePositions) {
            level.setBlockAndUpdate(pos, Blocks.AIR.defaultBlockState());
        }
        state.obstaclePositions.clear();
    }

    // ------------------------------------------------------------------
    // Bonemeal / growth tick
    // ------------------------------------------------------------------

    /**
     * Apply bonemeal to the crop at cropPos: advance growth stage by 1–2.
     * Returns true if growth happened, false if already max age.
     */
    public static boolean applyBonemeal(ServerLevel level, EpisodeState state, int cropIndex) {
        if (cropIndex < 0 || cropIndex >= state.farmingCropPositions.size()) return false;
        BlockPos pos   = state.farmingCropPositions.get(cropIndex);
        var block = level.getBlockState(pos);
        if (!(block.getBlock() instanceof CropBlock crop)) return false;
        if (crop.isMaxAge(block)) return false;

        int curAge = crop.getAge(block);
        int maxAge = crop.getMaxAge();
        int newAge = Math.min(maxAge, curAge + 1 + (int)(Math.random() * 2));
        level.setBlockAndUpdate(pos, crop.getStateForAge(newAge));
        if (cropIndex < state.cropGrowthStages.size()) {
            state.cropGrowthStages.set(cropIndex, newAge);
        }
        return true;
    }

    // ------------------------------------------------------------------
    // Marker placement
    // ------------------------------------------------------------------

    public static void placeNavigationMarker(ServerLevel level, EpisodeState state) {
        int mx = (int) Math.floor(state.targetX);
        int gy = getReferenceGroundY(level, (int) state.targetX, (int) state.targetZ);
        int mz = (int) Math.floor(state.targetZ);
        BlockPos markerPos = new BlockPos(mx, gy + 1, mz);
        level.setBlockAndUpdate(markerPos, Blocks.GOLD_BLOCK.defaultBlockState());
        state.markerPos = markerPos;
    }

    // ------------------------------------------------------------------
    // Y-resolution helpers — single implementation, three access points
    // ------------------------------------------------------------------

    /**
     * Returns the Y coordinate of the top solid surface block at (x, z).
     * This is the Y of the block itself — the agent stands at groundY + 1.
     */
    public static double resolveGroundY(ServerLevel level, double x, double z) {
        return (double) getReferenceGroundY(level, (int) Math.round(x), (int) Math.round(z));
    }

    /**
     * Returns the Y coordinate where the agent should stand at (x, z).
     * Equals resolveGroundY(x, z) + 1.
     */
    public static double resolveStandY(ServerLevel level, double x, double z) {
        return resolveGroundY(level, x, z) + 1.0;
    }

    /**
     * Returns the integer Y of the top solid surface block.
     * Use this for block placement (e.g., obstacle Y = groundY + 1).
     */
    public static int getReferenceGroundY(ServerLevel level, int x, int z) {
        BlockPos top = level.getHeightmapPos(
                Heightmap.Types.MOTION_BLOCKING_NO_LEAVES, new BlockPos(x, 0, z));
        // getHeightmapPos returns the first non-solid block above the surface,
        // so the surface block itself is at top.getY() - 1.
        return top.getY() - 1;
    }
}
