package com.aberrada.rlnpc;

import com.mojang.logging.LogUtils;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.event.entity.EntityAttributeCreationEvent;
import net.minecraftforge.event.server.ServerStartedEvent;
import net.minecraftforge.event.server.ServerStoppingEvent;
import net.minecraftforge.eventbus.api.IEventBus;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.javafmlmod.FMLJavaModLoadingContext;
import org.slf4j.Logger;

@Mod(RLNpcMod.MOD_ID)
public class RLNpcMod {
    public static final String MOD_ID = "rlnpc";
    public static final Logger LOGGER = LogUtils.getLogger();

    private static EnvironmentManager environmentManager;
    private static BridgeServer bridgeServer;

    public RLNpcMod() {
        IEventBus modBus = FMLJavaModLoadingContext.get().getModEventBus();
        ModEntities.ENTITY_TYPES.register(modBus);
        modBus.addListener(this::onEntityAttributes);

        MinecraftForge.EVENT_BUS.addListener(this::onServerStarted);
        MinecraftForge.EVENT_BUS.addListener(this::onServerStopping);

        LOGGER.info("RL NPC mod initialized.");
    }

    private void onEntityAttributes(EntityAttributeCreationEvent event) {
        event.put(ModEntities.RL_NPC.get(), RLNpcEntity.createAttributes().build());
    }

    private void onServerStarted(ServerStartedEvent event) {
        environmentManager = new EnvironmentManager(event.getServer());
        bridgeServer = new BridgeServer(environmentManager);
        bridgeServer.start();
        LOGGER.info("RL environment ready. HTTP bridge listening on http://127.0.0.1:8765");
    }

    private void onServerStopping(ServerStoppingEvent event) {
        if (bridgeServer != null) {
            bridgeServer.stop();
            bridgeServer = null;
        }
        environmentManager = null;
        LOGGER.info("RL environment shut down.");
    }

    public static EnvironmentManager getEnvironmentManager() {
        return environmentManager;
    }
}