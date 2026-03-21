package com.aberrada.rlnpc;

import net.minecraft.core.BlockPos;
import net.minecraft.world.level.block.CropBlock;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.Vec3;

import java.util.Locale;

public class ObservationBuilder {

    public static double[] build(RLNpcEntity agent, EpisodeState state) {
        double dx = state.targetX - agent.getX();
        double dz = state.targetZ - agent.getZ();
        double distance = Math.sqrt(dx * dx + dz * dz);

        double yawNorm = normalizeYaw(agent.getYRot()) / 180.0;
        double blockedFront = isBlockedFront(agent) ? 1.0 : 0.0;
        double onGround = agent.onGround() ? 1.0 : 0.0;
        double stuckNorm = Math.min(state.stuckSteps / 10.0, 1.0);
        double taskId = state.taskId;

        double cropInFront = isMatureCropInFront(agent) ? 1.0 : 0.0;
        double nearTarget = distance <= 1.10 ? 1.0 : 0.0;

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
                nearTarget
        };
    }

    private static double normalizeYaw(float yaw) {
        double result = yaw;
        while (result <= -180.0) result += 360.0;
        while (result > 180.0) result -= 360.0;
        return result;
    }

    private static Vec3 horizontalLook(RLNpcEntity agent) {
        Vec3 look = agent.getLookAngle();
        Vec3 flat = new Vec3(look.x, 0.0, look.z);

        if (flat.lengthSqr() < 1e-8) {
            return new Vec3(0.0, 0.0, 1.0);
        }

        return flat.normalize();
    }

    public static BlockPos frontBlockPos(RLNpcEntity agent) {
        Vec3 look = horizontalLook(agent);
        Vec3 origin = agent.position().add(0.0, 0.2, 0.0);
        Vec3 front = origin.add(look.scale(0.9));
        return BlockPos.containing(front.x, origin.y, front.z);
    }

    public static BlockPos frontCropPos(RLNpcEntity agent) {
        Vec3 look = horizontalLook(agent);
        Vec3 origin = agent.position().add(0.0, 0.5, 0.0);
        Vec3 front = origin.add(look.scale(0.9));
        return BlockPos.containing(front.x, origin.y, front.z);
    }

    public static boolean isMatureCropInFront(RLNpcEntity agent) {
        BlockPos cropPos = frontCropPos(agent);
        BlockState block = agent.level().getBlockState(cropPos);
        return block.getBlock() instanceof CropBlock cropBlock && cropBlock.isMaxAge(block);
    }

    private static boolean isBlockedFront(RLNpcEntity agent) {
        BlockPos feetFront = frontBlockPos(agent);
        BlockPos headFront = feetFront.above();

        BlockState feetState = agent.level().getBlockState(feetFront);
        BlockState headState = agent.level().getBlockState(headFront);

        boolean feetBlocked = !feetState.isAir() && feetState.blocksMotion();
        boolean headBlocked = !headState.isAir() && headState.blocksMotion();

        return feetBlocked || headBlocked;
    }

    public static String obsToJson(double[] obs) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < obs.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(String.format(Locale.US, "%.6f", obs[i]));
        }
        sb.append("]");
        return sb.toString();
    }
}