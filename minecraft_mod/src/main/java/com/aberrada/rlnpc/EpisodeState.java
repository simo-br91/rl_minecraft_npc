package com.aberrada.rlnpc;

import net.minecraft.core.BlockPos;

import java.util.UUID;

public class EpisodeState {
    public int episodeStep = 0;
    public int maxSteps = 100;
    public double prevDistance = 0.0;
    public int stuckSteps = 0;
    public int invalidActionCount = 0;
    public boolean done = false;
    public boolean success = false;

    public String taskName = "navigation";
    public double taskId = 0.0;

    public double targetX = 8.0;
    public double targetY = 4.0;
    public double targetZ = 0.0;

    public UUID agentUuid = null;
    public BlockPos markerPos = null;
    public BlockPos farmingSoilPos = null;
    public BlockPos farmingCropPos = null;
    public boolean lastInteractValid = false;

    public void setTask(String task) {
        if ("farming".equalsIgnoreCase(task)) {
            this.taskName = "farming";
            this.taskId = 1.0;
            this.maxSteps = 120;
        } else {
            this.taskName = "navigation";
            this.taskId = 0.0;
            this.maxSteps = 100;
        }
    }

    public void reset(double initialDistance) {
        this.episodeStep = 0;
        this.prevDistance = initialDistance;
        this.stuckSteps = 0;
        this.invalidActionCount = 0;
        this.done = false;
        this.success = false;
        this.lastInteractValid = false;
    }
}
