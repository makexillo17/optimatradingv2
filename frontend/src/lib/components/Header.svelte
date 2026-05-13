<script lang="ts">
	import StatusOrbit from './StatusOrbit.svelte';
	import IgnitionSwitch from './IgnitionSwitch.svelte';

	type SystemStatus = 'IDLE' | 'RUNNING' | 'ERROR' | 'RECONNECTING';

	interface HeaderProps {
		systemStatus?: SystemStatus;
		onIgnition?: () => Promise<void> | void;
	}

	let {
		systemStatus = 'IDLE',
		onIgnition = () => {}
	}: HeaderProps = $props();
</script>

<header
	class="col-span-full flex items-center justify-between gap-6 rounded-[var(--radius-card)] border border-[var(--color-null-border)] bg-[var(--color-null-surface)] px-5 py-4"
	id="header-main"
>
	<!-- Left: Branding + Status -->
	<div class="flex shrink-0 items-center gap-4">
		<div class="flex flex-col">
			<h1
				class="font-data text-lg font-bold tracking-[0.3em] text-[var(--color-null-text)]"
				style="animation: flicker 8s ease-in-out infinite;"
			>
				𝔑𝔘𝔏𝔏
			</h1>
			<span class="text-[9px] font-medium uppercase tracking-[0.25em] text-[var(--color-null-text-ghost)]">
				OPTIMA TRADING V2
			</span>
		</div>
		<div class="mx-2 h-8 w-px bg-[var(--color-null-border)]"></div>
		<StatusOrbit status={systemStatus} />
	</div>

	<!-- Right: Ignition Switch -->
	<div class="w-full max-w-xs">
		<IgnitionSwitch onToggle={onIgnition} />
	</div>
</header>
