<script lang="ts">
	import BentoCard from './BentoCard.svelte';
	import { appState } from '$lib/state/app.svelte';
	import type { TradePoint } from '$lib/state/app.svelte';

	// ── Chart Data ──
	let equityData = $derived(appState.simulationResults.equityCurve ?? []);
	let trades = $derived(appState.simulationResults.trades ?? []);
	let hasData = $derived(equityData.length > 1);

	// ── Chart Dimensions ──
	const CHART_W = 600;
	const CHART_H = 200;
	const PAD = { top: 15, right: 15, bottom: 25, left: 50 };
	const plotW = CHART_W - PAD.left - PAD.right;
	const plotH = CHART_H - PAD.top - PAD.bottom;

	// ── Scale Helpers ──
	let dataMax = $derived(hasData ? Math.max(...equityData) : 100);
	let dataMin = $derived(hasData ? Math.min(...equityData) : 0);
	let dataRange = $derived(dataMax - dataMin || 1);

	function scaleX(i: number): number {
		if (equityData.length <= 1) return PAD.left;
		return PAD.left + (i / (equityData.length - 1)) * plotW;
	}

	function scaleY(val: number): number {
		return PAD.top + plotH - ((val - dataMin) / dataRange) * plotH;
	}

	// ── SVG Path ──
	let linePath = $derived.by(() => {
		if (!hasData) return '';
		return equityData
			.map((v, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(i).toFixed(1)} ${scaleY(v).toFixed(1)}`)
			.join(' ');
	});

	// Area fill path (line + close to bottom)
	let areaPath = $derived.by(() => {
		if (!hasData) return '';
		const bottom = PAD.top + plotH;
		return (
			linePath +
			` L ${scaleX(equityData.length - 1).toFixed(1)} ${bottom}` +
			` L ${scaleX(0).toFixed(1)} ${bottom} Z`
		);
	});

	// ── Y-axis ticks ──
	let yTicks = $derived.by(() => {
		const count = 5;
		const step = dataRange / (count - 1);
		return Array.from({ length: count }, (_, i) => {
			const val = dataMin + step * i;
			return { val, y: scaleY(val) };
		});
	});

	// ── Crosshair / Tooltip ──
	let hoveredIndex = $state<number | null>(null);
	let svgEl: SVGSVGElement | undefined = $state();

	let hoveredX = $derived(hoveredIndex !== null ? scaleX(hoveredIndex) : 0);
	let hoveredY = $derived(hoveredIndex !== null ? scaleY(equityData[hoveredIndex]) : 0);
	let hoveredBalance = $derived(hoveredIndex !== null ? equityData[hoveredIndex] : 0);

	// Calculate drawdown at hovered point
	let hoveredDrawdown = $derived.by(() => {
		if (hoveredIndex === null || !hasData) return 0;
		let peak = equityData[0];
		for (let i = 1; i <= hoveredIndex; i++) {
			peak = Math.max(peak, equityData[i]);
		}
		return peak > 0 ? ((peak - equityData[hoveredIndex]) / peak) * 100 : 0;
	});

	function handlePointerMove(e: PointerEvent | TouchEvent) {
		if (!svgEl || !hasData) return;
		const rect = svgEl.getBoundingClientRect();
		const clientX = 'touches' in e ? e.touches[0].clientX : (e as PointerEvent).clientX;
		const relX = clientX - rect.left;
		const svgX = (relX / rect.width) * CHART_W;
		const dataX = Math.round(((svgX - PAD.left) / plotW) * (equityData.length - 1));
		hoveredIndex = Math.max(0, Math.min(equityData.length - 1, dataX));
	}

	function handlePointerLeave() {
		hoveredIndex = null;
	}

	// ── Trade point selection ──
	function selectTrade(trade: TradePoint) {
		appState.selectedTradeIndex = trade.index;
		// Trade detail would be populated by the backend or fetched
		appState.selectedTradeDetail = null; // Clear until detail arrives
	}

	// Find nearest trade to hovered index
	let nearestTrade = $derived.by(() => {
		if (hoveredIndex === null || trades.length === 0) return null;
		let closest: TradePoint | null = null;
		let minDist = Infinity;
		for (const t of trades) {
			const d = Math.abs(t.index - hoveredIndex);
			if (d < minDist && d < 5) {
				minDist = d;
				closest = t;
			}
		}
		return closest;
	});
</script>

<BentoCard title="Equity Curve" colSpan={2} rowSpan={2} variant={hasData ? 'accent' : 'default'}>
	{#if hasData}
		<!-- Chart SVG -->
		<svg
			bind:this={svgEl}
			viewBox="0 0 {CHART_W} {CHART_H}"
			class="h-full w-full cursor-crosshair touch-none select-none"
			preserveAspectRatio="none"
			role="img"
			aria-label="Equity curve chart"
			onpointermove={handlePointerMove}
			ontouchmove={handlePointerMove}
			onpointerleave={handlePointerLeave}
			ontouchend={handlePointerLeave}
		>
			<defs>
				<!-- Area gradient -->
				<linearGradient id="equity-fill" x1="0" y1="0" x2="0" y2="1">
					<stop offset="0%" stop-color="var(--color-null-accent)" stop-opacity="0.15" />
					<stop offset="100%" stop-color="var(--color-null-accent)" stop-opacity="0" />
				</linearGradient>
				<!-- Line glow -->
				<filter id="line-glow">
					<feGaussianBlur stdDeviation="2" result="blur" />
					<feMerge>
						<feMergeNode in="blur" />
						<feMergeNode in="SourceGraphic" />
					</feMerge>
				</filter>
			</defs>

			<!-- Y-axis ticks -->
			{#each yTicks as tick}
				<line
					x1={PAD.left} y1={tick.y}
					x2={CHART_W - PAD.right} y2={tick.y}
					stroke="rgba(255,255,255,0.04)"
					stroke-width="0.5"
				/>
				<text
					x={PAD.left - 6} y={tick.y + 3}
					fill="rgba(255,255,255,0.2)"
					font-size="8"
					font-family="var(--font-mono)"
					text-anchor="end"
				>
					{tick.val >= 1000 ? (tick.val / 1000).toFixed(1) + 'k' : tick.val.toFixed(0)}
				</text>
			{/each}

			<!-- Zero line if it's in range -->
			{#if dataMin < 0}
				<line
					x1={PAD.left} y1={scaleY(0)}
					x2={CHART_W - PAD.right} y2={scaleY(0)}
					stroke="rgba(255,255,255,0.1)"
					stroke-width="0.5"
					stroke-dasharray="4,4"
				/>
			{/if}

			<!-- Area fill -->
			<path d={areaPath} fill="url(#equity-fill)" />

			<!-- Glow line (behind main) -->
			<path
				d={linePath}
				fill="none"
				stroke="var(--color-null-accent)"
				stroke-width="4"
				opacity="0.15"
				stroke-linecap="round"
				stroke-linejoin="round"
				style="filter: blur(3px);"
			/>

			<!-- Main line -->
			<path
				d={linePath}
				fill="none"
				stroke="var(--color-null-accent)"
				stroke-width="1.5"
				stroke-linecap="round"
				stroke-linejoin="round"
				filter="url(#line-glow)"
			/>

			<!-- Trade markers -->
			{#each trades as trade}
				{@const tx = scaleX(trade.index)}
				{@const ty = scaleY(trade.balance)}
				{@const isSelected = appState.selectedTradeIndex === trade.index}
				<g
					style="cursor: pointer;"
					onclick={() => selectTrade(trade)}
					onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') selectTrade(trade); }}
					role="button"
					tabindex="0"
					aria-label="Trade {trade.direction} {trade.result}"
				>
					<!-- Marker circle -->
					<circle
						cx={tx} cy={ty} r={isSelected ? 5 : 3}
						fill={trade.result === 'WIN' ? 'var(--color-null-accent)' : 'var(--color-null-veto)'}
						opacity={isSelected ? 1 : 0.7}
						stroke={isSelected ? 'white' : 'none'}
						stroke-width={isSelected ? 1 : 0}
					/>
					{#if isSelected}
						<circle
							cx={tx} cy={ty} r="8"
							fill="none"
							stroke={trade.result === 'WIN' ? 'var(--color-null-accent)' : 'var(--color-null-veto)'}
							stroke-width="0.5"
							opacity="0.5"
						/>
					{/if}
				</g>
			{/each}

			<!-- Crosshair -->
			{#if hoveredIndex !== null}
				<!-- Vertical line -->
				<line
					x1={hoveredX} y1={PAD.top}
					x2={hoveredX} y2={PAD.top + plotH}
					stroke="rgba(255,255,255,0.2)"
					stroke-width="0.5"
					stroke-dasharray="3,3"
				/>
				<!-- Horizontal line -->
				<line
					x1={PAD.left} y1={hoveredY}
					x2={CHART_W - PAD.right} y2={hoveredY}
					stroke="rgba(255,255,255,0.1)"
					stroke-width="0.5"
					stroke-dasharray="3,3"
				/>
				<!-- Point -->
				<circle
					cx={hoveredX} cy={hoveredY} r="4"
					fill="var(--color-null-accent)"
					stroke="var(--color-null-black)"
					stroke-width="1.5"
				/>
				<circle
					cx={hoveredX} cy={hoveredY} r="7"
					fill="none"
					stroke="var(--color-null-accent)"
					stroke-width="0.5"
					opacity="0.4"
				/>
			{/if}
		</svg>

		<!-- Tooltip overlay -->
		{#if hoveredIndex !== null}
			<div
				class="pointer-events-none absolute z-20 rounded-md border border-[var(--color-null-border-hover)] bg-[var(--color-null-surface)] px-3 py-2 shadow-xl"
				style="top: 2.5rem; right: 1rem;"
			>
				<div class="flex items-center gap-4">
					<div class="flex flex-col">
						<span class="text-[7px] uppercase text-[var(--color-null-text-ghost)]">Balance</span>
						<span class="font-data text-sm font-bold text-[var(--color-null-accent)]">
							${hoveredBalance.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
						</span>
					</div>
					<div class="h-6 w-px bg-[var(--color-null-border)]"></div>
					<div class="flex flex-col">
						<span class="text-[7px] uppercase text-[var(--color-null-text-ghost)]">Drawdown</span>
						<span class="font-data text-sm font-bold text-[var(--color-null-veto)]">
							-{hoveredDrawdown.toFixed(2)}%
						</span>
					</div>
					{#if nearestTrade}
						<div class="h-6 w-px bg-[var(--color-null-border)]"></div>
						<div class="flex flex-col">
							<span class="text-[7px] uppercase text-[var(--color-null-text-ghost)]">Trade</span>
							<span class="font-data text-[10px] font-semibold"
								class:text-[var(--color-null-accent)]={nearestTrade.result === 'WIN'}
								class:text-[var(--color-null-veto)]={nearestTrade.result === 'LOSS'}
							>
								{nearestTrade.direction} {nearestTrade.result} ({nearestTrade.pnl > 0 ? '+' : ''}{nearestTrade.pnl.toFixed(2)}%)
							</span>
						</div>
					{/if}
				</div>
			</div>
		{/if}

		<!-- Bottom stats bar -->
		<div class="mt-3 flex items-center justify-between border-t border-[var(--color-null-border)] pt-2">
			<span class="font-data text-[9px] text-[var(--color-null-text-ghost)]">
				{equityData.length} candles
			</span>
			<div class="flex gap-4">
				<span class="font-data text-[9px] text-[var(--color-null-text-ghost)]">
					MAX: <span class="text-[var(--color-null-accent)]">${dataMax.toFixed(0)}</span>
				</span>
				<span class="font-data text-[9px] text-[var(--color-null-text-ghost)]">
					MIN: <span class="text-[var(--color-null-veto)]">${dataMin.toFixed(0)}</span>
				</span>
			</div>
		</div>
	{:else}
		<!-- Empty state -->
		<div class="flex h-52 flex-col items-center justify-center gap-3">
			<svg class="h-12 w-12 text-[var(--color-null-text-ghost)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
				<polyline points="22,12 18,12 15,21 9,3 6,12 2,12" />
			</svg>
			<span class="text-[10px] uppercase tracking-[0.2em] text-[var(--color-null-text-ghost)]">
				EJECUTAR BACKTEST PARA GENERAR CURVA
			</span>
		</div>
	{/if}
</BentoCard>
