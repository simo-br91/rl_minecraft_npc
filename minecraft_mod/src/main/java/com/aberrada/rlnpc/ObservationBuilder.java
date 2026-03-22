package com.aberrada.rlnpc;

import net.minecraft.core.BlockPos;
import net.minecraft.world.level.block.CropBlock;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.Vec3;

import java.util.Locale;

public class ObservationBuilder {

    /**
     * Builds a 11-element observation vector for the agent.
     *
     * Index  Feature
     * -----  -------
     *   0    dx          (target_x - agent_x)
     *   1    dz          (target_z - agent_z)
     *   2    distance    (Euclidean XZ distance to target)
     *   3    yaw_norm    (yaw in [-1, 1])
     *   4    blocked_front  (1 if wall directly ahead at feet level)
     *   5    on_ground   (1 if solid block directly below)
     *   6    stuck_norm  (stuck steps / 10, clamped to [0, 1])
     *   7    task_id     (0 = navigation, 1 = farming)
     *   8    crop_in_front  (1 if mature crop directly ahead)
     *   9    near_target    (1 if distance ≤ 1.10)
     *  10    obstacle_1block_ahead  (1 if exactly 1-block-high wall ahead = jumpable)
     */
    public static double[] build(RLNpcEntity agent, EpisodeState state) {
        double dx       = state.targetX - agent.getX();
        double dz       = state.targetZ - agent.getZ();
        double distance = Math.sqrt(dx * dx + dz * dz);

        double yawNorm        = normalizeYaw(agent.getYRot()) / 180.0;
        double blockedFront   = isBlockedFront(agent) ? 1.0 : 0.0;
        double onGround       = isOnGround(agent) ? 1.0 : 0.0;
        double stuckNorm      = Math.min(state.stuckSteps / 10.0, 1.0);
        double taskId         = state.taskId;
        double cropInFront    = isMatureCropInFront(agent) ? 1.0 : 0.0;
        double nearTarget     = distance <= 1.10 ? 1.0 : 0.0;
        double obstacle1Block = is1BlockObstacleAhead(agent) ? 1.0 : 0.0;

        return new double[] {
                dx,
                dz,
                distance,
                yawNorm,
                blockedFront,
                onGround,
                stuckNorm,
                taskId,
                cropInFront,
                nearTarget,
                obstacle1Block
        };
    }

    // -----------------------------------------------------------------------
    // Feature helpers
    // -----------------------------------------------------------------------

    private static double normalizeYaw(float yaw) {
        double r = yaw;
        while (r <= -180.0) r += 360.0;
        while (r > 180.0)   r -= 360.0;
        return r;
    }

    /**
     * True if there is a solid block directly in front of the agent at feet OR
     * head level (i.e. the agent cannot walk forward).
     */
    private static boolean isBlockedFront(RLNpcEntity agent) {
        BlockPos feet = ActionExecutor.frontBlockPos(agent);
        BlockPos head = feet.above();
        BlockState fs = agent.level().getBlockState(feet);
        BlockState hs = agent.level().getBlockState(head);
        return (!fs.isAir() && fs.blocksMotion()) || (!hs.isAir() && hs.blocksMotion());
    }

    /**
     * True if the block directly below the agent is solid.
     * More reliable than agent.onGround() for a teleport-driven entity.
     */
    public static boolean isOnGround(RLNpcEntity agent) {
        BlockPos below = BlockPos.containing(agent.getX(), agent.getY() - 0.1, agent.getZ());
        BlockState bs  = agent.level().getBlockState(below);
        return !bs.isAir() && bs.blocksMotion();
    }

    /**
     * True when there is a jumpable (exactly 1-block-high) obstacle directly
     * ahead: blocked at feet level but clear one block higher.
     * This gives the policy a direct cue to use the jump action.
     */
    public static boolean is1BlockObstacleAhead(RLNpcEntity agent) {
        Vec3 flat = ActionExecutor.getHorizontalLook(agent);
        if (flat == null) return false;

        double targetX = agent.getX() + flat.x * 0.9;
        double targetZ = agent.getZ() + flat.z * 0.9;
        double y = agent.getY();

        boolean blockedLow  = ActionExecutor.isBlockedAt(agent, targetX, y,       targetZ);
        boolean blockedHigh = ActionExecutor.isBlockedAt(agent, targetX, y + 1.0, targetZ);
        return blockedLow && !blockedHigh;
    }

    // -----------------------------------------------------------------------
    // Crop helpers (farming task)
    // -----------------------------------------------------------------------

    public static BlockPos frontCropPos(RLNpcEntity agent) {
        Vec3 flat = ActionExecutor.getHorizontalLook(agent);
        if (flat == null) flat = new Vec3(0, 0, 1);
        Vec3 origin = agent.position().add(0.0, 0.5, 0.0);
        Vec3 front  = origin.add(flat.scale(0.9));
        return BlockPos.containing(front.x, origin.y, front.z);
    }

    public static boolean isMatureCropInFront(RLNpcEntity agent) {
        BlockPos cropPos  = frontCropPos(agent);
        BlockState block  = agent.level().getBlockState(cropPos);
        return block.getBlock() instanceof CropBlock cropBlock && cropBlock.isMaxAge(block);
    }

    // -----------------------------------------------------------------------
    // Serialisation
    // -----------------------------------------------------------------------

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
