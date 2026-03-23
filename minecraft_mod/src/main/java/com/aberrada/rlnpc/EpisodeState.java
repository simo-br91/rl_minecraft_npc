package com.aberrada.rlnpc;

import net.minecraft.core.BlockPos;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

public class EpisodeState {
    public int     episodeStep        = 0;
    public int     maxSteps           = 100;
    public double  prevDistance       = 0.0;
    public int     stuckSteps         = 0;
    public int     invalidActionCount = 0;
    public boolean done               = false;
    public boolean success            = false;

    public String  taskName    = "navigation";
    public double  taskId      = 0.0;

    public double  targetX     = 8.0;
    public double  targetY     = 4.0;
    public double  targetZ     = 0.0;

    // ---- curriculum / experiment config (set on each reset) ----
    /** When true, only terminal rewards are given (no distance shaping). */
    public boolean sparseReward           = false;
    /** Curriculum: minimum XZ distance to target. -1 = use task default. */
    public double  curriculumMinDist      = -1.0;
    /** Curriculum: maximum XZ distance to target. -1 = use task default. */
    public double  curriculumMaxDist      = -1.0;
    /** Curriculum: exact number of obstacles to place. -1 = use task default (1-2). */
    public int     curriculumNumObstacles = -1;

    public UUID     agentUuid       = null;
    public BlockPos markerPos       = null;
    public BlockPos farmingSoilPos  = null;
    public BlockPos farmingCropPos  = null;

    /** Obstacle blocks placed for the current navigation episode. */
    public List<BlockPos> obstaclePositions = new ArrayList<>();

    /** Per-step flags set by ActionExecutor / interact handler. */
    public boolean lastInteractValid  = false;
    public boolean lastJumpedObstacle = false;

    public void setTask(String task) {
        if ("farming".equalsIgnoreCase(task)) {
            this.taskName = "farming";
            this.taskId   = 1.0;
            this.maxSteps = 200;
        } else {
            this.taskName = "navigation";
            this.taskId   = 0.0;
            this.maxSteps = 150;
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
    }
}
