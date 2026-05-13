<script lang="ts">
	import './layout.css';
	import { onMount } from 'svelte';
	import { initializeApp, appState, toggleSystem } from '$lib/state/app.svelte';
	import Header from '$lib/components/Header.svelte';

	let { children } = $props();

	let connectionStatus = $derived(
		appState.connectionStatus === 'CONNECTED'
			? 'RUNNING' as const
			: appState.connectionStatus === 'RECONNECTING'
				? 'RECONNECTING' as const
				: appState.systemStatus === 'ERROR'
					? 'ERROR' as const
					: 'IDLE' as const
	);

	onMount(() => {
		initializeApp();
	});
</script>

<svelte:head>
	<title>𝔑𝔘𝔏𝔏 — OptimaTrading V2</title>
</svelte:head>

<div class="flex min-h-dvh flex-col bg-[var(--color-null-black)]">
	<!-- Scanline effect overlay -->
	<div
		class="pointer-events-none fixed inset-0 z-50 opacity-[0.015]"
		style="background: repeating-linear-gradient(
			0deg,
			transparent,
			transparent 2px,
			rgba(255,255,255,0.03) 2px,
			rgba(255,255,255,0.03) 4px
		);"
	></div>

	<!-- Bento Grid -->
	<main class="bento-grid flex-1 {appState.forensicMode ? 'relative' : ''}">
		<!-- Forensic Overlay Tint -->
		{#if appState.forensicMode}
			<div class="pointer-events-none absolute inset-0 z-10 bg-yellow-900/5 mix-blend-color"></div>
			<div class="pointer-events-none absolute inset-0 z-10 flex items-center justify-center opacity-5">
				<span class="rotate-[-30deg] font-data text-[120px] font-bold tracking-widest text-[var(--color-null-accent)] whitespace-nowrap">
					MODO FORENSE: {appState.forensicTimestamp ? new Date(appState.forensicTimestamp).toLocaleTimeString('es-MX', { hour12: false }) : ''}
				</span>
			</div>
		{/if}

		<Header
			systemStatus={connectionStatus}
			onIgnition={toggleSystem}
		/>

		{@render children()}
	</main>

	<!-- Footer -->
	<footer class="flex items-center justify-between border-t border-[var(--color-null-border)] px-6 py-3">
		<span class="font-data text-[9px] tracking-[0.15em] text-[var(--color-null-text-ghost)]">
			NULL.v0.1.0 // OPTIMA TRADING V2
		</span>
		<div class="flex items-center gap-3">
			{#if appState.telemetry.latencyMs !== null}
				<span class="font-data text-[9px] text-[var(--color-null-text-ghost)]">
					{appState.telemetry.latencyMs}ms
				</span>
			{/if}
			<span class="font-data text-[9px] text-[var(--color-null-text-ghost)]">
				{new Date().toISOString().slice(0, 19).replace('T', ' ')}
			</span>
		</div>
	</footer>
</div>
