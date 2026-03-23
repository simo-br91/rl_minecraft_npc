package com.aberrada.rlnpc;

import net.minecraft.world.entity.EntityType;
import net.minecraft.world.entity.Mob;
import net.minecraft.world.entity.PathfinderMob;
import net.minecraft.world.entity.ai.attributes.AttributeSupplier;
import net.minecraft.world.entity.ai.attributes.Attributes;
import net.minecraft.world.level.Level;

/**
 * The controllable RL NPC entity.
 *
 * - No vanilla AI goals (controlled entirely via the HTTP bridge)
 * - Can take damage from mobs (invulnerable=false)
 * - Has a food bar simulated in EpisodeState
 * - Full movement speed range for walk/sprint
 * - Respawns next episode (via getOrCreateAgent in EnvironmentManager)
 */
public class RLNpcEntity extends PathfinderMob {

    public RLNpcEntity(EntityType<? extends PathfinderMob> type, Level level) {
        super(type, level);
        this.setNoAi(true);
        this.setInvulnerable(false);  // Takes damage from mobs
    }

    public static AttributeSupplier.Builder createAttributes() {
        return Mob.createMobAttributes()
                .add(Attributes.MAX_HEALTH,       20.0)
                .add(Attributes.MOVEMENT_SPEED,   0.3)   // base walk speed
                .add(Attributes.FOLLOW_RANGE,     24.0)
                .add(Attributes.ARMOR,            6.0)   // iron armor equivalent
                .add(Attributes.ARMOR_TOUGHNESS,  2.0)
                .add(Attributes.ATTACK_DAMAGE,    6.0)   // iron sword base damage
                .add(Attributes.ATTACK_SPEED,     1.6)
                .add(Attributes.KNOCKBACK_RESISTANCE, 0.0);
    }

    @Override
    protected void registerGoals() {
        // No vanilla AI — RL controls this entity
    }

    @Override
    public boolean removeWhenFarAway(double distanceToClosestPlayer) {
        return false;  // Never despawn
    }

    /**
     * Allow the entity to be hurt by mobs (important for combat learning).
     * The entity is NOT invulnerable so mobs can kill it.
     */
    @Override
    public boolean hurt(net.minecraft.world.damagesource.DamageSource source, float amount) {
        boolean result = super.hurt(source, amount);
        if (result) {
            RLNpcMod.LOGGER.debug("RL NPC took {} damage, health now {}", amount, this.getHealth());
        }
        return result;
    }

    @Override
    public boolean isPushable() {
        return true;  // Mobs can push the agent
    }
}