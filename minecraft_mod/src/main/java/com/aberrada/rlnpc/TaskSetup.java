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
 */
public class TaskSetup {

    private static final int    MAX_CROPS          = 10;
    private static final double NAV_DEFAULT_MIN    = 7.0;
    private static final double NAV_DEFAULT_MAX    = 14.0;

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

        // Scale maxSteps with distance
        int base = (int)(dist * 15);
        state.maxSteps = Math.max(100, Math.min(300, base));
    }

    public static void placeNavigationObstacles(ServerLevel level, EpisodeState state,
                                                 double spawnX, double spawnZ, Random rng) {
        int count = state.curriculumNumObstacles >= 0
                ? state.curriculumNumObstacles
                : 1 + rng.nextInt(2);
        if (count == 0) return;

        double dx   = state.targetX - spawnX;
        double dz   = state.targetZ - spawnZ;
        double dist = Math.sqrt(dx * dx + dz * dz);
        if (dist < 4.0) return;

        double[] fracs = {0.30, 0.50, 0.65, 0.80};
        int placed = 0;
        for (int i = 0; i < fracs.length && placed < count; i++) {
            double frac = fracs[i] + (rng.nextDouble() - 0.5) * 0.06;
            int ox = (int) Math.round(spawnX + dx * frac);
            int oz = (int) Math.round(spawnZ + dz * frac);
            int gy = getReferenceGroundY(level, ox, oz);

            // Avoid target and spawn
            double wallDistTarget = Math.sqrt(Math.pow(ox - state.targetX, 2)
                                            + Math.pow(oz - state.targetZ, 2));
            double wallDistSpawn  = Math.sqrt(Math.pow(ox - spawnX, 2)
                                            + Math.pow(oz - spawnZ, 2));
            if (wallDistTarget < 2.0 || wallDistSpawn < 2.0) continue;

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
     * @param numCrops      How many wheat plots to create (1–MAX_CROPS).
     * @param fullCycle     If true, place dirt+farmland and seeds (not pre-grown).
     *                      If false, place pre-grown wheat immediately.
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
        int centerX = 6 + rng.nextInt(5) - 2;
        int centerZ = 2 + rng.nextInt(5) - 2;

        List<int[]> positions = generateCropPositions(centerX, centerZ, numCrops, rng);
        int groundY = getReferenceGroundY(level, centerX, centerZ);

        for (int[] pos : positions) {
            int cx = pos[0], cz = pos[1];
            BlockPos soilPos = new BlockPos(cx, groundY, cz);
            BlockPos cropPos = soilPos.above();

            // Place farmland
            level.setBlockAndUpdate(soilPos, Blocks.FARMLAND.defaultBlockState());

            if (fullCycle) {
                // Plant seeds (age=0), agent must use bonemeal or wait
                CropBlock wheat = (CropBlock) Blocks.WHEAT;
                level.setBlockAndUpdate(cropPos, wheat.getStateForAge(0));
                state.cropGrowthStages.add(0);
            } else {
                // Pre-grown wheat
                CropBlock wheat = (CropBlock) Blocks.WHEAT;
                level.setBlockAndUpdate(cropPos, wheat.getStateForAge(wheat.getMaxAge()));
                state.cropGrowthStages.add(wheat.getMaxAge());
            }

            state.farmingSoilPositions.add(soilPos);
            state.farmingCropPositions.add(cropPos);
            state.cropHarvested.add(false);
        }

        // Set initial target to nearest crop from origin
        if (!state.farmingCropPositions.isEmpty()) {
            BlockPos first = state.farmingCropPositions.get(0);
            state.targetX = first.getX() + 0.5;
            state.targetY = first.getY();
            state.targetZ = first.getZ() + 0.5;
            state.activeCropIndex = 0;
        }

        state.maxSteps = 150 + numCrops * 80;
    }

    /** Generate distinct crop positions spread around the center. */
    private static List<int[]> generateCropPositions(int cx, int cz, int n, Random rng) {
        List<int[]> result = new ArrayList<>();
        // Use a grid layout with random jitter
        int cols = (int) Math.ceil(Math.sqrt(n));
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
                level.setBlockAndUpdate(cropPos,  Blocks.AIR.defaultBlockState());
                level.setBlockAndUpdate(soilPos,  Blocks.GRASS_BLOCK.defaultBlockState());
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
     * Apply bonemeal to the crop at cropPos: advance growth stage by 1–3.
     * Returns true if growth happened, false if already max age.
     */
    public static boolean applyBonemeal(ServerLevel level, EpisodeState state, int cropIndex) {
        if (cropIndex < 0 || cropIndex >= state.farmingCropPositions.size()) return false;
        BlockPos pos = state.farmingCropPositions.get(cropIndex);
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
        int gy = (int) Math.floor(resolveGroundY(level, state.targetX, state.targetZ));
        int mz = (int) Math.floor(state.targetZ);
        BlockPos markerPos = new BlockPos(mx, gy + 1, mz);
        level.setBlockAndUpdate(markerPos, Blocks.GOLD_BLOCK.defaultBlockState());
        state.markerPos = markerPos;
    }

    // ------------------------------------------------------------------
    // Y resolution helpers (single implementation)
    // ------------------------------------------------------------------

    /**
     * Returns the Y coordinate of the top solid surface block at (x,z).
     * Agent should stand at groundY + 1.
     */
    public static double resolveGroundY(ServerLevel level, double x, double z) {
        BlockPos top = level.getHeightmapPos(
                Heightmap.Types.MOTION_BLOCKING_NO_LEAVES,
                new BlockPos((int) Math.round(x), 0, (int) Math.round(z)));
        return top.getY() - 1.0;
    }

    /** Returns Y the agent should be positioned at (standing on top of surface). */
    public static double resolveStandY(ServerLevel level, double x, double z) {
        return resolveGroundY(level, x, z) + 1.0;
    }

    /** Integer version for block placement. */
    public static int getReferenceGroundY(ServerLevel level, int x, int z) {
        BlockPos top = level.getHeightmapPos(
                Heightmap.Types.MOTION_BLOCKING_NO_LEAVES, new BlockPos(x, 0, z));
        return top.getY() - 1;
    }
}