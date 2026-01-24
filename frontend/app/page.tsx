"use client";

import React, { useState, useEffect } from "react";
import GlassCard from "@/components/ui/GlassCard";
import { AreaChart, Area, ResponsiveContainer, YAxis } from "recharts";
import { Activity, Zap, Layers, Command, BarChart3, Settings, ShieldAlert, Cpu } from "lucide-react";
import { motion } from "framer-motion";

// --- DUMMY DATA ---
const BTC_DATA = Array.from({ length: 40 }, (_, i) => ({
    time: i,
    value: 45000 + Math.random() * 1000 + (i * 50),
}));

const TICKERS = [
    { symbol: "BTC-USDT", price: "47,234.50", change: "+1.2%", status: "up" },
    { symbol: "ETH-USDT", price: "2,432.10", change: "-0.5%", status: "down" },
    { symbol: "SOL-USDT", price: "98.45", change: "+4.3%", status: "up" },
    { symbol: "BNB-USDT", price: "320.12", change: "+0.1%", status: "up" },
    { symbol: "ADA-USDT", price: "0.5432", change: "-1.2%", status: "down" },
    { symbol: "XRP-USDT", price: "0.6210", change: "+0.0%", status: "neutral" },
];

const MODULE_STATUS = {
    regime: "TRENDING",
    confidence: 94,
    active_modules: ["SMC_ICT", "Carry_Trade"],
    risk_level: "LOW"
};

// --- SUB-COMPONENTS (Inline for compactness as requested in single page integration) ---

const TrendVisor = () => (
    <div className="h-full w-full flex flex-col justify-between p-4">
        <div className="flex justify-between items-start">
            <div>
                <h3 className="text-text-secondary text-xs uppercase tracking-widest font-semibold font-mono">Market Trend</h3>
                <div className="flex items-baseline gap-2 mt-1">
                    <span className="text-3xl font-bold font-mono tracking-tighter text-text-primary">47,234.50</span>
                    <span className="text-signal-up text-sm font-mono">+1.24%</span>
                </div>
            </div>
            <div className="flex gap-2">
                <span className="bg-signal-up/20 text-signal-up text-[10px] px-2 py-0.5 rounded-full font-mono uppercase">Strong Buy</span>
            </div>
        </div>

        <div className="flex-1 min-h-[100px] mt-4 -mx-4 -mb-4">
            <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={BTC_DATA}>
                    <defs>
                        <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#10B981" stopOpacity={0.1} />
                            <stop offset="95%" stopColor="#10B981" stopOpacity={0} />
                        </linearGradient>
                    </defs>
                    <YAxis domain={['auto', 'auto']} hide />
                    <Area
                        type="monotone"
                        dataKey="value"
                        stroke="#10B981"
                        strokeWidth={1.5}
                        fillOpacity={1}
                        fill="url(#colorValue)"
                    />
                </AreaChart>
            </ResponsiveContainer>
        </div>
    </div>
);

const RadarModule = () => (
    <div className="p-4 h-full flex flex-col">
        <div className="flex items-center justify-between mb-4">
            <h3 className="text-text-secondary text-xs uppercase tracking-widest font-semibold flex items-center gap-2">
                <Activity size={14} /> Radar
            </h3>
            <span className="w-2 h-2 rounded-full bg-accent-primary animate-pulse"></span>
        </div>
        <div className="flex-1 overflow-y-auto space-y-1 pr-2 scrollbar-none">
            {TICKERS.map((t) => (
                <div key={t.symbol} className="flex justify-between items-center p-2 rounded-lg hover:bg-white/5 transition-colors group cursor-pointer">
                    <span className="text-sm font-mono text-text-secondary group-hover:text-text-primary transition-colors">{t.symbol}</span>
                    <div className="flex gap-4 text-sm font-mono">
                        <span>{t.price}</span>
                        <span className={t.status === 'up' ? 'text-signal-up' : t.status === 'down' ? 'text-signal-down' : 'text-text-secondary'}>
                            {t.change}
                        </span>
                    </div>
                </div>
            ))}
        </div>
    </div>
);

const SystemStatus = () => (
    <div className="p-4 h-full flex flex-col">
        <h3 className="text-text-secondary text-xs uppercase tracking-widest font-semibold mb-4 flex items-center gap-2">
            <Cpu size={14} /> Algorithm State
        </h3>

        <div className="grid grid-cols-2 gap-3 mb-4">
            <div className="bg-white/5 rounded-lg p-3 border border-white/5">
                <span className="text-text-secondary text-[10px] uppercase block mb-1">Regime</span>
                <span className="text-accent-primary font-bold tracking-wider">{MODULE_STATUS.regime}</span>
            </div>
            <div className="bg-white/5 rounded-lg p-3 border border-white/5">
                <span className="text-text-secondary text-[10px] uppercase block mb-1">Confidence</span>
                <span className="text-signal-up font-bold font-mono">{MODULE_STATUS.confidence}%</span>
            </div>
        </div>

        <div className="flex-1">
            <span className="text-text-secondary text-[10px] uppercase block mb-2">Active Modules</span>
            <div className="flex flex-wrap gap-2">
                {MODULE_STATUS.active_modules.map(m => (
                    <span key={m} className="text-xs border border-white/10 px-2 py-1 rounded bg-white/5 text-text-primary">
                        {m}
                    </span>
                ))}
            </div>
        </div>
    </div>
);

const CommandCenter = () => {
    const actions = [
        { icon: Zap, label: "Sniper" },
        { icon: Layers, label: "Modules" },
        { icon: BarChart3, label: "Analytics" },
        { icon: ShieldAlert, label: "Risk" },
        { icon: Settings, label: "Config" },
    ];

    return (
        <div className="flex items-center gap-2 p-1.5 bg-black/50 backdrop-blur-xl border border-white/10 rounded-full mx-auto shadow-2xl">
            {actions.map((Action, i) => (
                <button
                    key={i}
                    className="p-3 rounded-full text-text-secondary hover:text-text-primary hover:bg-white/10 transition-all active:scale-95 relative group"
                >
                    <Action.icon size={20} />
                    <span className="absolute -top-10 left-1/2 -translate-x-1/2 bg-black border border-white/10 px-2 py-1 rounded text-[10px] opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                        {Action.label}
                    </span>
                </button>
            ))}
            <div className="w-px h-6 bg-white/10 mx-1"></div>
            <button className="flex items-center gap-2 px-4 py-2 bg-accent-primary hover:bg-accent-primary/90 text-white rounded-full text-xs font-semibold transition-colors shadow-lg shadow-accent-primary/20">
                <Command size={14} /> <span>Execute</span>
            </button>
        </div>
    );
};

// --- MAIN PAGE ---

export default function Dashboard() {
    return (
        <main className="min-h-screen bg-background p-6 font-sans flex flex-col relative">
            {/* Header */}
            <header className="flex justify-between items-center mb-8">
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-accent-primary rounded-lg flex items-center justify-center font-bold text-white">
                        O
                    </div>
                    <div>
                        <h1 className="text-lg font-bold tracking-tight text-white">OptimaTrading<span className="text-accent-primary">V2</span></h1>
                        <p className="text-text-secondary text-xs">Institutional Terminal</p>
                    </div>
                </div>
                <div className="flex items-center gap-4 text-xs font-mono text-text-secondary">
                    <span>EST: 14:02:32</span>
                    <span className="w-2 h-2 rounded-full bg-signal-up"></span>
                    <span>SYSTEM ONLINE</span>
                </div>
            </header>

            {/* BENTO GRID LAYOUT */}
            <div className="flex-1 grid grid-cols-12 grid-rows-6 gap-4 pb-20">

                {/* 1. Main Chart (TrendVisor) - Large Area */}
                <GlassCard className="col-span-8 row-span-4" active>
                    <TrendVisor />
                </GlassCard>

                {/* 2. Radar Module - Side Panel */}
                <GlassCard className="col-span-4 row-span-4">
                    <RadarModule />
                </GlassCard>

                {/* 3. System Status - Bottom Area */}
                <GlassCard className="col-span-4 row-span-2">
                    <SystemStatus />
                </GlassCard>

                {/* 4. Filler / Logs / Other Metrics */}
                <GlassCard className="col-span-8 row-span-2 p-4">
                    <div className="flex items-center justify-between mb-3">
                        <h3 className="text-text-secondary text-xs uppercase tracking-widest font-semibold">Live Logs</h3>
                        <span className="text-[10px] text-text-secondary font-mono bg-white/5 px-2 py-1 rounded">Auto-Scroll</span>
                    </div>
                    <div className="font-mono text-xs space-y-2 text-text-secondary overflow-hidden h-full mask-linear-fade">
                        <p><span className="text-white/30">[14:00:01]</span> Checking Gap Sniper...</p>
                        <p><span className="text-white/30">[14:00:05]</span> Market Regime: <span className="text-accent-primary">TRENDING</span></p>
                        <p><span className="text-white/30">[14:00:12]</span> Consent Score: 0.82 (Confident)</p>
                        <p><span className="text-white/30">[14:00:15]</span> <span className="text-signal-up">BUY SIGNAL</span> detected on BTC-USDT</p>
                    </div>
                </GlassCard>

            </div>

            {/* Command Center (Floating Bottom) */}
            <div className="fixed bottom-8 left-1/2 -translate-x-1/2 z-50">
                <CommandCenter />
            </div>

        </main>
    );
}
