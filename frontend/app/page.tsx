export default function Dashboard() {
    return (
        <main className="min-h-screen bg-black text-white p-8 font-sans">
            <h1 className="text-3xl font-bold mb-8">OptimaTrading V2 - Style Verification</h1>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Panel 1 */}
                <div className="border border-zinc-800 p-6 rounded-xl bg-zinc-900/50">
                    <h2 className="text-xl font-semibold mb-2 text-accent-primary">Panel 1</h2>
                    <p className="text-gray-400">Verificando estilos básicos.</p>
                </div>

                {/* Gráfico */}
                <div className="border border-zinc-800 p-6 rounded-xl bg-zinc-900/50">
                    <h2 className="text-xl font-semibold mb-2 text-green-500">Gráfico</h2>
                    <div className="h-20 bg-green-500/10 rounded flex items-center justify-center border border-green-500/20">
                        [Placeholder Visual]
                    </div>
                </div>

                {/* Status */}
                <div className="border border-zinc-800 p-6 rounded-xl bg-zinc-900/50">
                    <h2 className="text-xl font-semibold mb-2 text-blue-500">Status</h2>
                    <p className="font-mono text-sm text-gray-300">SYSTEM: ONLINE</p>
                </div>
            </div>
        </main>
    );
}
