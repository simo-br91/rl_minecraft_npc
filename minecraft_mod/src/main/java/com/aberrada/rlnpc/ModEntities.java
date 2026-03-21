package com.aberrada.rlnpc;

import net.minecraft.world.entity.EntityType;
import net.minecraft.world.entity.MobCategory;
import net.minecraftforge.registries.DeferredRegister;
import net.minecraftforge.registries.ForgeRegistries;
import net.minecraftforge.registries.RegistryObject;

public class ModEntities {
    public static final DeferredRegister<EntityType<?>> ENTITY_TYPES =
            DeferredRegister.create(ForgeRegistries.ENTITY_TYPES, RLNpcMod.MOD_ID);

    public static final RegistryObject<EntityType<RLNpcEntity>> RL_NPC =
            ENTITY_TYPES.register("rl_npc",
                    () -> EntityType.Builder.<RLNpcEntity>of(RLNpcEntity::new, MobCategory.CREATURE)
                            .sized(0.6f, 1.8f)
                            .clientTrackingRange(8)
                            .build(RLNpcMod.MOD_ID + ":rl_npc"));
}