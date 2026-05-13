<script lang="ts">
	import { appState } from '$lib/state/app.svelte';

	// The timeline covers the last 24 hours
	const HOURS_24_MS = 24 * 60 * 60 * 1000;

	let isDragging = $state(false);
	let scrubValue = $state(100); // 0 to 100 percentage
	let sliderEl: HTMLInputElement | undefined = $state();

	// Computed real time
	let currentTime = $derived(Date.now());
	let timelineStart = $derived(currentTime - HOURS_24_MS);
	
	// Value 100 = LIVE
	let isLive = $derived(scrubValue === 100);

	// The timestamp calculated from the slider position
	let scrubTimestamp = $derived(
		isLive ? currentTime : timelineStart + (HOURS_24_MS * (scrubValue / 100))
	);

	function updateForensicState() {
		if (isLive) {
			appState.forensicMode = false;
			appState.forensicTimestamp = null;
		} else {
			appState.forensicMode = true;
			appState.forensicTimestamp = scrubTimestamp;
		}
	}

	function handleInput(e: Event) {
		const target = e.target as HTMLInputElement;
		scrubValue = parseFloat(target.value);
		updateForensicState();
	}

	function handlePointerDown() {
		isDragging = true;
	}

	function handlePointerUp() {
		isDragging = false;
	}

	function returnToLive() {
		scrubValue = 100;
		updateForensicState();
	}

	// Formatting the time for the tooltip
	function formatTime(ts: number) {
		const d = new Date(ts);
		return d.toLocaleTimeString('es-MX', { hour12: false, hour: '2-digit', minute: '2-digit' }) + ' hrs';
	}
</script>

<!-- Floating "Regreso al Presente" Button -->
{#if !isLive}
	<button
		class="fixed bottom-24 right-6 z-50 rounded-full bg-[var(--color-null-accent)] px-6 py-3 font-data text-[11px] font-bold uppercase tracking-widest text-black shadow-[0_0_20px_rgba(163,230,53,0.3)] transition-transform hover:scale-105 active:scale-95"
		onclick={returnToLive}
		aria-label="Regreso al Presente"
	>
		&#9654; RETORNAR A LIVE
	</button>
{/if}

<!-- Timeline Scrubber Container -->
<div class="fixed bottom-0 left-0 right-0 z-40 border-t border-[var(--color-null-border)] bg-[var(--color-null-black)]/90 px-6 py-4 backdrop-blur-md">
	<div class="mx-auto flex w-full max-w-7xl items-center gap-4">
		
		<!-- Start label (-24h) -->
		<span class="font-data text-[10px] text-[var(--color-null-text-ghost)]">
			-24H
		</span>

		<!-- Scrubber -->
		<div class="relative flex-1">
			<input
				bind:this={sliderEl}
				type="range"
				min="0"
				max="100"
				step="0.1"
				value={scrubValue}
				oninput={handleInput}
				onpointerdown={handlePointerDown}
				onpointerup={handlePointerUp}
				onpointerleave={handlePointerUp}
				class="scrubber w-full"
				aria-label="Forensic Time Scrubber"
			/>
			
			<!-- Hover/Drag Tooltip -->
			<div
				class="absolute bottom-full mb-3 -ml-12 w-24 rounded-md border border-[var(--color-null-border)] px-2 py-1 text-center font-data text-[10px] transition-opacity duration-200"
				class:opacity-100={isDragging || !isLive}
				class:opacity-0={!isDragging && isLive}
				style="left: {scrubValue}%;"
				class:bg-[var(--color-null-accent)]={isLive}
				class:text-black={isLive}
				class:bg-[var(--color-null-surface)]={!isLive}
				class:text-[var(--color-null-text)]={!isLive}
			>
				{#if isLive}
					<span class="font-bold">LIVE</span>
				{:else}
					<span>{formatTime(scrubTimestamp)}</span>
				{/if}
				<!-- Pointer arrow -->
				<div
					class="absolute -bottom-[5px] left-1/2 -ml-[5px] h-0 w-0 border-l-[5px] border-r-[5px] border-t-[5px] border-transparent"
					class:border-t-[var(--color-null-accent)]={isLive}
					class:border-t-[var(--color-null-surface)]={!isLive}
				></div>
			</div>
		</div>

		<!-- End label (LIVE) -->
		<span class="font-data text-[10px] font-bold" class:text-[var(--color-null-accent)]={isLive} class:text-[var(--color-null-text-ghost)]={!isLive}>
			LIVE
		</span>
	</div>
</div>

<style>
	/* Forensic Scrubber Custom Styles */
	.scrubber {
		-webkit-appearance: none;
		appearance: none;
		background: transparent;
		cursor: pointer;
		position: relative;
		z-index: 10;
	}

	.scrubber::before {
		content: '';
		position: absolute;
		top: 50%;
		left: 0;
		right: 0;
		height: 2px;
		margin-top: -1px;
		background: #27272a;
		border-radius: 1px;
		z-index: -1;
	}

	.scrubber::-webkit-slider-thumb {
		-webkit-appearance: none;
		appearance: none;
		width: 14px;
		height: 24px;
		background: #ffffff;
		border-radius: 2px;
		border: 1px solid #000;
		box-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
		transition: transform 0.1s ease;
	}

	.scrubber::-webkit-slider-thumb:active {
		transform: scaleX(1.3);
		background: var(--color-null-accent);
		box-shadow: 0 0 15px rgba(163, 230, 53, 0.4);
	}

	.scrubber::-moz-range-thumb {
		width: 14px;
		height: 24px;
		background: #ffffff;
		border-radius: 2px;
		border: 1px solid #000;
		box-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
		transition: transform 0.1s ease;
	}

	.scrubber::-moz-range-thumb:active {
		transform: scaleX(1.3);
		background: var(--color-null-accent);
		box-shadow: 0 0 15px rgba(163, 230, 53, 0.4);
	}
</style>
