<script lang="ts">
	import BentoCard from './BentoCard.svelte';
	import { appState, getActiveConviction } from '$lib/state/app.svelte';

	// ── Reactive Conviction ──
	let conviction = $derived(getActiveConviction());

	// Smooth animation via CSS transition on the arc
	let arcAngle = $derived(conviction * 1.8); // 0-100 → 0-180 degrees

	// Dynamic color thresholds
	let gaugeColor = $derived(
		conviction > 70
			? 'var(--color-null-accent)'
			: conviction > 40
				? '#a1a1aa' // zinc-400
				: 'var(--color-null-veto)'
	);

	let gaugeColorClass = $derived(
		conviction > 70
			? 'text-[var(--color-null-accent)]'
			: conviction > 40
				? 'text-zinc-400'
				: 'text-[var(--color-null-veto)]'
	);

	let glowOpacity = $derived(conviction > 70 ? 0.3 : conviction > 40 ? 0.1 : 0.2);

	// SVG Arc path calculation for semi-circle
	// Arc from 180° (left) to 0° (right), centered at (100, 100), radius 80
	function describeArc(cx: number, cy: number, r: number, startAngle: number, endAngle: number): string {
		const start = polarToCartesian(cx, cy, r, endAngle);
		const end = polarToCartesian(cx, cy, r, startAngle);
		const largeArcFlag = endAngle - startAngle <= 180 ? 0 : 1;
		return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 0 ${end.x} ${end.y}`;
	}

	function polarToCartesian(cx: number, cy: number, r: number, angleDeg: number) {
		const rad = ((angleDeg - 180) * Math.PI) / 180;
		return {
			x: cx + r * Math.cos(rad),
			y: cy + r * Math.sin(rad)
		};
	}

	// Background track (full semi-circle)
	let trackPath = $derived(describeArc(100, 95, 75, 0, 180));
	// Active arc (conviction portion)
	let activePath = $derived(describeArc(100, 95, 75, 0, Math.max(0.5, arcAngle)));

	// Needle endpoint
	let needleEnd = $derived(polarToCartesian(100, 95, 65, arcAngle));

	// Status label
	let statusLabel = $derived(
		conviction > 85
			? 'CONVICCIÓN EXTREMA'
			: conviction > 70
				? 'SEÑAL FUERTE'
				: conviction > 40
					? 'NEUTRAL'
					: conviction > 20
						? 'SEÑAL DÉBIL'
						: 'SIN SEÑAL'
	);
</script>

<BentoCard title="Radar de Convicción" colSpan={2} rowSpan={2}>
	<div class="flex flex-col items-center justify-center py-2">
		<!-- SVG Gauge -->
		<svg viewBox="0 0 200 120" class="h-48 w-full max-w-xs">
			<defs>
				<!-- Glow filter -->
				<filter id="gauge-glow" x="-20%" y="-20%" width="140%" height="140%">
					<feGaussianBlur stdDeviation="4" result="blur" />
					<feMerge>
						<feMergeNode in="blur" />
						<feMergeNode in="SourceGraphic" />
					</feMerge>
				</filter>
				<!-- Gradient for track -->
				<linearGradient id="track-grad" x1="0" y1="0" x2="1" y2="0">
					<stop offset="0%" stop-color="var(--color-null-veto)" stop-opacity="0.15" />
					<stop offset="40%" stop-color="#a1a1aa" stop-opacity="0.15" />
					<stop offset="70%" stop-color="var(--color-null-accent)" stop-opacity="0.15" />
				</linearGradient>
			</defs>

			<!-- Tick marks -->
			{#each [0, 20, 40, 60, 80, 100] as tick}
				{@const pos = polarToCartesian(100, 95, 82, tick * 1.8)}
				{@const innerPos = polarToCartesian(100, 95, 75, tick * 1.8)}
				<line
					x1={innerPos.x} y1={innerPos.y}
					x2={pos.x} y2={pos.y}
					stroke="rgba(255,255,255,0.2)"
					stroke-width="1"
				/>
				{@const labelPos = polarToCartesian(100, 95, 90, tick * 1.8)}
				<text
					x={labelPos.x} y={labelPos.y}
					fill="rgba(255,255,255,0.2)"
					font-size="7"
					font-family="var(--font-mono)"
					text-anchor="middle"
					dominant-baseline="middle"
				>
					{tick}
				</text>
			{/each}

			<!-- Track (background arc) -->
			<path
				d={trackPath}
				fill="none"
				stroke="url(#track-grad)"
				stroke-width="8"
				stroke-linecap="round"
			/>

			<!-- Active Arc -->
			<path
				d={activePath}
				fill="none"
				stroke={gaugeColor}
				stroke-width="8"
				stroke-linecap="round"
				filter="url(#gauge-glow)"
				style="transition: d 0.6s cubic-bezier(0.16, 1, 0.3, 1), stroke 0.4s ease;"
			/>

			<!-- Glow behind arc -->
			<path
				d={activePath}
				fill="none"
				stroke={gaugeColor}
				stroke-width="14"
				stroke-linecap="round"
				opacity={glowOpacity}
				style="filter: blur(6px); transition: d 0.6s cubic-bezier(0.16, 1, 0.3, 1);"
			/>

			<!-- Needle -->
			<line
				x1="100" y1="95"
				x2={needleEnd.x} y2={needleEnd.y}
				stroke={gaugeColor}
				stroke-width="1.5"
				stroke-linecap="round"
				opacity="0.8"
				style="transition: x2 0.6s cubic-bezier(0.16, 1, 0.3, 1), y2 0.6s cubic-bezier(0.16, 1, 0.3, 1);"
			/>
			<!-- Needle center dot -->
			<circle cx="100" cy="95" r="3" fill={gaugeColor} opacity="0.9"
				style="transition: fill 0.4s ease;"
			/>

			<!-- Center Value -->
			<text
				x="100" y="82"
				fill={gaugeColor}
				font-size="28"
				font-family="var(--font-mono)"
				font-weight="700"
				text-anchor="middle"
				dominant-baseline="middle"
				style="transition: fill 0.4s ease;"
			>
				{conviction}
			</text>
			<text
				x="100" y="95"
				fill="rgba(255,255,255,0.3)"
				font-size="6"
				font-family="var(--font-mono)"
				text-anchor="middle"
				letter-spacing="0.15em"
			>
				% BAYESIANO
			</text>
		</svg>

		<!-- Status Label -->
		<div class="mt-1 flex items-center gap-2">
			<div
				class="h-1.5 w-1.5 rounded-full"
				style="background-color: {gaugeColor}; transition: background-color 0.4s ease;"
			></div>
			<span class="font-data text-[10px] font-semibold uppercase tracking-[0.2em] {gaugeColorClass}">
				{statusLabel}
			</span>
		</div>

		<!-- Substatus -->
		<span class="mt-1 font-data text-[8px] tracking-[0.1em] text-[var(--color-null-text-ghost)]">
			{appState.connectionStatus === 'CONNECTED'
				? 'TELEMETRÍA EN VIVO'
				: appState.connectionStatus === 'RECONNECTING'
					? '⟳ REINTENTANDO...'
					: 'ESPERANDO DATOS'}
		</span>
	</div>
</BentoCard>
