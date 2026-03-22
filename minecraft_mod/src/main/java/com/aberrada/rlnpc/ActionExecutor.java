package com.aberrada.rlnpc;

import net.minecraft.core.BlockPos;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.Vec3;

public class ActionExecutor {
    private static final float TURN_DEGREES = 15.0f;
    private static final double FORWARD_STEP = 0.35;

    // 0 forward, 1 turn_left, 2 turn_right, 3 interact (handled elsewhere), 4 no_op
    public static boolean applyMovementAction(RLNpcEntity agent, int action) {
        return switch (action) {
            case 0 -> moveForward(agent);
            case 1 -> turnLeft(agent);
            case 2 -> turnRight(agent);
            case 4 -> true;
            case 5 -> jump(agent);
            default -> false;
        };
    }

    private static boolean moveForward(RLNpcEntity agent) {
        Vec3 look = agent.getLookAngle();
        Vec3 flat = new Vec3(look.x, 0.0, look.z);

        if (flat.lengthSqr() < 1e-8) {
            return false;
        }

        flat = flat.normalize();
        double targetX = agent.getX() + flat.x * FORWARD_STEP;
        double targetZ = agent.getZ() + flat.z * FORWARD_STEP;
        double targetY = agent.getY();

        if (isBlockedAt(agent, targetX, targetY, targetZ)) {
            return false;
        }

        float yaw = agent.getYRot();
        agent.moveTo(targetX, targetY, targetZ, yaw, 0.0f);
        syncRotations(agent, yaw);
        return true;
    }

    private static boolean turnLeft(RLNpcEntity agent) {
        return rotateInPlace(agent, agent.getYRot() - TURN_DEGREES);
    }

    private static boolean turnRight(RLNpcEntity agent) {
        return rotateInPlace(agent, agent.getYRot() + TURN_DEGREES);
    }

    private static boolean rotateInPlace(RLNpcEntity agent, float newYaw) {
        agent.moveTo(agent.getX(), agent.getY(), agent.getZ(), newYaw, 0.0f);
        syncRotations(agent, newYaw);
        return true;
    }

    private static boolean jump(RLNpcEntity agent) {
        if (!agent.onGround()) {
            return false;   // can't jump mid-air
        }
        agent.setDeltaMovement(
            agent.getDeltaMovement().x,
            0.42,           // same impulse vanilla uses
            agent.getDeltaMovement().z
        );
        agent.hasImpulse = true;  // tells the client to sync the motion packet
        return true;
    }

    private static void syncRotations(RLNpcEntity agent, float yaw) {
        agent.setYRot(yaw);
        agent.setYHeadRot(yaw);
        agent.yBodyRot = yaw;
        agent.yHeadRot = yaw;
    }

    private static boolean isBlockedAt(RLNpcEntity agent, double x, double y, double z) {
        BlockPos feetPos = BlockPos.containing(x, y, z);
        BlockPos headPos = feetPos.above();

        BlockState feet = agent.level().getBlockState(feetPos);
        BlockState head = agent.level().getBlockState(headPos);

        boolean feetBlocked = !feet.isAir() && feet.blocksMotion();
        boolean headBlocked = !head.isAir() && head.blocksMotion();

        return feetBlocked || headBlocked;
    }
}