"use client";

import React, { useEffect, useState } from "react";
import { AreaChart, Area, ResponsiveContainer, YAxis, Tooltip } from "recharts";
import { Activity, Zap, Layers, Command, BarChart3, Settings, ShieldAlert, Cpu, Terminal, TrendingUp, RefreshCw } from "lucide-react";
import { BentoGrid, BentoGridItem } from "@/components/ui/bento-grid";
import GlassCard from "@/components/ui/GlassCard";

// --- INTERFACES ---
interface MarketData {
    time: string;
    close: number;
    ema50: number | null;
    ema200: number | null;
}

interface MarketData {
    time: string;
    close: number;
    ema50: number | null;
    ema200: number | null;
}

interface ApiResponse {
    symbol: string;
    price: number;
    change: number;
    data: MarketData[];
}

interface Verdict {
    recommendation: string;
    confidence: number;
    justification: string;
}

interface Signal {
    timestamp: string;
    asset: string;
    signal: string;
}

interface RenderData {
    verdict: Verdict | null;
    history: Signal[];
}

// --- SUB-COMPONENTS ---

const TrendVisor = ({ data, loading }: { data: ApiResponse | null, loading: boolean }) => {
    if (loading || !data) {
        return <div className="animate-pulse flex items-center justify-center h-full text-text-secondary font-mono text-xs">Loading Live Data...</div>;
    }

    const isUp = data.change >= 0;
    const color = isUp ? "#10B981" : "#F43F5E";

    return (
        <div className="h-full w-full flex flex-col justify-between">
            <div className="flex justify-between items-start">
                <div>
                    <div className="flex items-baseline gap-2 mt-1">
                        <span className="text-3xl font-bold font-mono tracking-tighter text-text-primary">
                            ${data.price.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                        </span>
                        <span className={`text-sm font-mono ${isUp ? 'text-signal-up' : 'text-signal-down'}`}>
                            {isUp ? '+' : ''}{data.change.toFixed(2)}%
                        </span>
                    </div>
                    <p className="text-xs text-text-secondary font-mono mt-1">BTC/USD (Kraken)</p>
                </div>
            </div>

            <div className="flex-1 min-h-[100px] mt-4 -mx-4 -mb-4 relative">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data.data}>
                        <defs>
                            <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={color} stopOpacity={0.1} />
                                <stop offset="95%" stopColor={color} stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <YAxis domain={['auto', 'auto']} hide />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#000', borderColor: '#333' }}
                            itemStyle={{ color: '#fff', fontSize: '12px', fontFamily: 'monospace' }}
                            labelStyle={{ display: 'none' }}
                        />
                        <Area
                            type="monotone"
                            dataKey="close"
                            stroke={color}
                            strokeWidth={2}
                            fillOpacity={1}
                            fill="url(#colorValue)"
                        />
                        {/* Can add EMAs here if we want more density, maybe togglable */}
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

const RadarModule = ({ history }: { history: Signal[] }) => (
    <div className="h-full flex flex-col overflow-hidden">
        <h4 className="text-xs text-text-secondary uppercase mb-2">Recent Signals (Render API)</h4>
        <div className="flex-1 overflow-y-auto space-y-1 scrollbar-none p-1">
            {history.length === 0 ? (
                <div className="text-center text-text-secondary text-xs italic mt-10">
                    Waiting for Signals...
                </div>
            ) : (
                history.map((sig, i) => (
                    <div key={i} className="flex justify-between items-center p-2 rounded bg-white/5 border border-white/5">
                        <span className="text-xs font-mono text-text-secondary">{new Date(sig.timestamp).toLocaleTimeString()}</span>
                        <span className={`text-xs font-bold ${sig.signal.toUpperCase().includes('BUY') ? 'text-signal-up' :
                            sig.signal.toUpperCase().includes('SELL') ? 'text-signal-down' : 'text-text-primary'
                            }`}>
                            {sig.signal}
                        </span>
                    </div>
                ))
            )}
        </div>
    </div>
);

const SystemStatus = ({ regime, verdict }: { regime: string, verdict: Verdict | null }) => (
    <div className="h-full flex flex-col pt-2">
        <div className="grid grid-cols-2 gap-3 mb-4">
            <div className="bg-white/5 rounded-lg p-3 border border-border-subtle">
                <span className="text-text-secondary text-[10px] uppercase block mb-1">Regime</span>
                <span className="text-accent-primary font-bold tracking-wider">{regime || 'Loading...'}</span>
            </div>
            <div className="bg-white/5 rounded-lg p-3 border border-border-subtle">
                <span className="text-text-secondary text-[10px] uppercase block mb-1">Confidence</span>
                <span className="text-signal-up font-bold font-mono">
                    {verdict ? (verdict.confidence * 100).toFixed(0) : 0}%
                </span>
            </div>
        </div>

        <div className="flex-1">
            <span className="text-text-secondary text-[10px] uppercase block mb-2">Verdict (Render)</span>
            <div className="p-2 bg-white/5 rounded text-xs text-text-primary mb-2">
                {verdict ? verdict.recommendation.toUpperCase() : "Establishing Link..."}
            </div>
            <p className="text-[10px] text-text-secondary line-clamp-2">
                {verdict?.justification || "Waiting for consensus..."}
            </p>
        </div>
    </div>
);

const LogsModule = () => (
    <div className="font-mono text-xs space-y-2 text-text-secondary overflow-hidden h-full mask-linear-fade">
        <p><span className="text-white/30">[SYS]</span> Data Feed Connected.</p>
        <p><span className="text-white/30">[SYS]</span> Optimizing route for 47k...</p>
    </div>
);

// --- MAIN PAGE ---

export default function Dashboard() {
    const [mounted, setMounted] = useState(false);
    const [marketData, setMarketData] = useState<ApiResponse | null>(null);
    const [renderData, setRenderData] = useState<RenderData>({ verdict: null, history: [] });
    const [loading, setLoading] = useState(true);
    const [lastUpdated, setLastUpdated] = useState<Date>(new Date());

    useEffect(() => {
        setMounted(true);
    }, []);

    const fetchData = async () => {
        try {
            // 1. Local Next.js API (Market Data + Indicators)
            const resMarket = await fetch('/api/market-data');
            const dataMarket = await resMarket.json();
            if (dataMarket.symbol) {
                setMarketData(dataMarket);
                setLastUpdated(new Date());
            }

            // 2. Remote Render API (Algorithm Verdict + History)
            // Using Promise.allSettled to not fail if Render is sleeping
            const RENDER_URL = "https://optimatradingv2.onrender.com";

            try {
                const resVerdict = await fetch(`${RENDER_URL}/analyze/BTCUSD`, { signal: AbortSignal.timeout(5000) });
                const verdict = await resVerdict.json();

                const resHistory = await fetch(`${RENDER_URL}/history`, { signal: AbortSignal.timeout(5000) });
                const historyData = await resHistory.json();

                setRenderData({
                    verdict: verdict || null,
                    history: historyData.history || []
                });
            } catch (err) {
                console.warn("Render API unreachable (might be sleeping):", err);
            }

        } catch (e) {
            console.error("Failed to fetch data", e);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchData();
        const interval = setInterval(fetchData, 60000); // Update every minute to respect rate limits roughly
        return () => clearInterval(interval);
    }, []);

    const items = [
        {
            title: "Market Trend Visor",
            description: "Real-time Kraken Data",
            header: <TrendVisor data={marketData} loading={loading} />,
            className: "md:col-span-2 md:row-span-2",
            icon: <TrendingUp className="h-4 w-4 text-neutral-500" />,
        },
        {
            title: "Radar",
            description: "Algorithmic Consensus",
            header: <RadarModule history={renderData.history} />,
            className: "md:col-span-1 md:row-span-2",
            icon: <Activity className="h-4 w-4 text-neutral-500" />,
        },
        {
            title: "System Status",
            description: "Regime & Confidence",
            header: <SystemStatus regime="TRENDING" verdict={renderData.verdict} />,
            className: "md:col-span-1 md:row-span-1",
            icon: <Cpu className="h-4 w-4 text-neutral-500" />,
        },
        {
            title: "Live Logs",
            description: "Execution Stream",
            header: <LogsModule />,
            className: "md:col-span-2 md:row-span-1",
            icon: <Terminal className="h-4 w-4 text-neutral-500" />,
        },
    ];

    if (!mounted) return <div className="min-h-screen bg-black" />;

    return (
        <main className="min-h-screen bg-background p-6 font-sans relative flex flex-col">
            {/* Header */}
            <header className="flex justify-between items-center mb-8 max-w-7xl mx-auto w-full">
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-accent-primary rounded-lg flex items-center justify-center font-bold text-white shadow-[0_0_15px_rgba(0,122,255,0.5)]">
                        O
                    </div>
                    <div>
                        <h1 className="text-lg font-bold tracking-tight text-white/90">OptimaTrading<span className="text-accent-primary">V2</span></h1>
                        <p className="text-text-secondary text-xs">Institutional Terminal</p>
                    </div>
                </div>
                <div className="flex items-center gap-4 text-xs font-mono text-text-secondary">
                    <button onClick={fetchData} className="hover:text-white transition-colors">
                        <RefreshCw size={12} className={loading ? "animate-spin" : ""} />
                    </button>
                    <span>UPDATED: {lastUpdated.toLocaleTimeString()}</span>
                    <span className="w-2 h-2 rounded-full bg-signal-up animate-pulse"></span>
                    <span>SYSTEM ONLINE</span>
                </div>
            </header>

            {/* BENTO GRID */}
            <BentoGrid className="max-w-7xl mx-auto w-full">
                {items.map((item, i) => (
                    <BentoGridItem
                        key={i}
                        title={item.title}
                        description={item.description}
                        header={item.header}
                        className={item.className}
                        icon={item.icon}
                    />
                ))}
            </BentoGrid>

            {/* Command Center (Floating Bottom) */}
            <div className="fixed bottom-8 left-1/2 -translate-x-1/2 z-50">
                <div className="flex items-center gap-2 p-1.5 bg-black/80 backdrop-blur-xl border border-border-subtle rounded-full mx-auto shadow-2xl">
                    <button className="p-3 rounded-full text-text-secondary hover:text-white transition-colors hover:bg-white/10">
                        <Zap size={20} />
                    </button>
                    <button className="p-3 rounded-full text-text-secondary hover:text-white transition-colors hover:bg-white/10">
                        <Layers size={20} />
                    </button>
                    <button className="p-3 rounded-full text-text-secondary hover:text-white transition-colors hover:bg-white/10">
                        <BarChart3 size={20} />
                    </button>
                    <div className="w-px h-6 bg-white/10 mx-1"></div>
                    <button className="flex items-center gap-2 px-4 py-2 bg-accent-primary hover:bg-accent-primary/90 text-white rounded-full text-xs font-semibold transition-colors shadow-lg shadow-accent-primary/20">
                        <Command size={14} /> <span>Execute</span>
                    </button>
                </div>
            </div>

        </main>
    );
}
