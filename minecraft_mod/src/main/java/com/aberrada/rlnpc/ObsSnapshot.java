package com.aberrada.rlnpc;

import net.minecraft.core.BlockPos;

import java.util.ArrayList;
import java.util.List;

/**
 * Immutable snapshot of the episode state fields needed by ObservationBuilder.
 *
 * This decouples ObservationBuilder from the mutable EpisodeState, allowing
 * observations to be built for logging or replay without a live episode.
 * (Fixes Issue 7.6 — ObservationBuilder coupled to mutable EpisodeState)
 */
public final class ObsSnapshot {

    public final double  targetX;
    public final double  targetZ;
    public final int     stuckSteps;
    public final double  taskId;
    public final float   health;
    public final int     foodLevel;
    public final int     activeSlot;
    public final int     cropsHarvested;
    public final int     totalCrops;
    public final List<BlockPos> farmingCropPositions;
    public final List<Boolean>  cropHarvested;

    public ObsSnapshot(
            double  targetX,
            double  targetZ,
            int     stuckSteps,
            double  taskId,
            float   health,
            int     foodLevel,
            int     activeSlot,
            int     cropsHarvested,
            int     totalCrops,
            List<BlockPos> farmingCropPositions,
            List<Boolean>  cropHarvested) {
        this.targetX              = targetX;
        this.targetZ              = targetZ;
        this.stuckSteps           = stuckSteps;
        this.taskId               = taskId;
        this.health               = health;
        this.foodLevel            = foodLevel;
        this.activeSlot           = activeSlot;
        this.cropsHarvested       = cropsHarvested;
        this.totalCrops           = totalCrops;
        // Defensive copies so the snapshot is truly immutable
        this.farmingCropPositions = List.copyOf(farmingCropPositions);
        this.cropHarvested        = List.copyOf(cropHarvested);
    }

    /** Convenience factory — create a snapshot from a live EpisodeState. */
    public static ObsSnapshot from(EpisodeState state) {
        return new ObsSnapshot(
                state.targetX,
                state.targetZ,
                state.stuckSteps,
                state.taskId,
                state.health,
                state.foodLevel,
                state.activeSlot,
                state.cropsHarvested,
                state.totalCrops,
                state.farmingCropPositions.isEmpty()
                        ? List.of()
                        : new ArrayList<>(state.farmingCropPositions),
                state.cropHarvested.isEmpty()
                        ? List.of()
                        : new ArrayList<>(state.cropHarvested)
        );
    }
}
