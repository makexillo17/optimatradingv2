/**
 * ═══════════════════════════════════════════════════════
 *  𝔑𝔘𝔏𝔏 — Global State Manager & API Client
 *  OptimaTrading V2 Frontend
 * ═══════════════════════════════════════════════════════
 *
 *  Singleton state management using Svelte 5 Runes.
 *  Connects Frontend (Vercel) ↔ Backend (Hostinger/Easypanel)
 *  via HTTPS (Fetch) for commands and WSS for live telemetry.
 */

// ── Configuration ──
const API_BASE = 'https://optimatrading-optima-app.af5gdr.easypanel.host';
const WSS_BASE = 'wss://optimatrading-optima-app.af5gdr.easypanel.host';

// ── Types ──
export type SystemStatus = 'IDLE' | 'RUNNING' | 'ERROR';
export type ConnectionStatus = 'DISCONNECTED' | 'CONNECTING' | 'CONNECTED' | 'RECONNECTING';

export interface Signal {
	id: string;
	symbol: string;
	direction: 'LONG' | 'SHORT';
	confidence: number;
	timestamp: number;
	source: string;
}

export interface SimulationResults {
	totalReturn?: number;
	sharpeRatio?: number;
	maxDrawdown?: number;
	winRate?: number;
	totalTrades?: number;
	equityCurve?: number[];
	trades?: TradePoint[];
	status?: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED';
}

export interface TradePoint {
	index: number;
	balance: number;
	drawdown: number;
	direction: 'LONG' | 'SHORT';
	result: 'WIN' | 'LOSS';
	pnl: number;
	symbol: string;
}

export interface TradeDetail {
	// Technical Inputs
	gapDetected: string;
	fibLevel: string;
	volumeFilter: string;
	smcSignal: string;
	liquidityZone: string;
	volatilityRegime: string;
	// Claude AI Verdict
	claudeVerdict: string;
	claudeConfidence: number;
	claudeReasoning: string;
	// Risk Calculation
	positionSize: string;
	kellyFraction: string;
	stopLoss: string;
	takeProfit: string;
	riskRewardRatio: string;
	atrMultiple: string;
}

export interface TelemetryState {
	connected: boolean;
	lastPing: number | null;
	retryCount: number;
	latencyMs: number | null;
}

export interface EngineState {
	id: string;
	name: string;
	status: 'ONLINE' | 'OFFLINE' | 'ERROR';
	weight: number;
	lastVerdict: string;
	lastUpdate: number;
	pulseActive: boolean;
}

export const ENGINE_DEFINITIONS: { id: string; name: string }[] = [
	{ id: 'smc', name: 'SMC / ICT' },
	{ id: 'liquidity', name: 'Liquidez' },
	{ id: 'volatility', name: 'Volatilidad' },
	{ id: 'sentiment', name: 'IA Sentiment' },
	{ id: 'gap_sniper', name: 'Gap Sniper' },
	{ id: 'carry_trade', name: 'Carry Trade' },
	{ id: 'momentum', name: 'Momentum' },
	{ id: 'mean_revert', name: 'Mean Revert' },
	{ id: 'orderflow', name: 'Order Flow' },
	{ id: 'bayesian', name: 'Bayesiano' }
];

export interface CalibrationConfig {
	aiWeight: number;
	smcSensitivity: number;
	riskThreshold: number;
}

export interface ForensicSnapshot {
	timestamp: number;
	conviction: number;
	engines: EngineState[];
}

interface AppStateShape {
	systemStatus: SystemStatus;
	currentSignals: Signal[];
	simulationResults: SimulationResults;
	telemetry: TelemetryState;
	connectionStatus: ConnectionStatus;
	logs: LogEntry[];
	conviction: number;
	engines: EngineState[];
	selectedTradeIndex: number | null;
	selectedTradeDetail: TradeDetail | null;
	calibration: CalibrationConfig;
	calibrationSyncing: boolean;
	forensicMode: boolean;
	forensicTimestamp: number | null;
	forensicSnapshots: ForensicSnapshot[];
}

export interface LogEntry {
	timestamp: number;
	level: 'INFO' | 'WARN' | 'ERROR';
	message: string;
	source: string;
}

// ── Global State ──
export const appState: AppStateShape = $state({
	systemStatus: 'IDLE',
	currentSignals: [],
	simulationResults: {},
	telemetry: {
		connected: false,
		lastPing: null,
		retryCount: 0,
		latencyMs: null
	},
	connectionStatus: 'DISCONNECTED',
	logs: [],
	conviction: 0,
	engines: ENGINE_DEFINITIONS.map((e) => ({
		...e,
		status: 'OFFLINE' as const,
		weight: 1.0,
		lastVerdict: '—',
		lastUpdate: 0,
		pulseActive: false
	})),
	selectedTradeIndex: null,
	selectedTradeDetail: null,
	calibration: {
		aiWeight: 1.0,
		smcSensitivity: 0.5,
		riskThreshold: 0.3
	},
	calibrationSyncing: false,
	forensicMode: false,
	forensicTimestamp: null,
	forensicSnapshots: []
});

// ── Derived State ──
// NOTE: $derived at module level requires Svelte 5 runes mode.
// Components should use $derived locally:
//   let ready = $derived(appState.telemetry.connected && appState.systemStatus !== 'ERROR');
//
// Export a helper function for non-rune contexts:
export function getIsSystemReady(): boolean {
	return appState.telemetry.connected && appState.systemStatus !== 'ERROR';
}

export function getActiveSignalCount(): number {
	return appState.currentSignals.length;
}

export function getActiveConviction(): number {
	if (appState.forensicMode && appState.forensicTimestamp !== null && appState.forensicSnapshots.length > 0) {
		const targetTs = appState.forensicTimestamp;
		// Find nearest snapshot
		let nearest = appState.forensicSnapshots[0];
		let minDiff = Math.abs(nearest.timestamp - targetTs);
		for (let i = 1; i < appState.forensicSnapshots.length; i++) {
			const diff = Math.abs(appState.forensicSnapshots[i].timestamp - targetTs);
			if (diff < minDiff) {
				minDiff = diff;
				nearest = appState.forensicSnapshots[i];
			}
		}
		return nearest.conviction;
	}
	return appState.conviction;
}

export function getActiveEngines(): EngineState[] {
	if (appState.forensicMode && appState.forensicTimestamp !== null && appState.forensicSnapshots.length > 0) {
		const targetTs = appState.forensicTimestamp;
		// Find nearest snapshot
		let nearest = appState.forensicSnapshots[0];
		let minDiff = Math.abs(nearest.timestamp - targetTs);
		for (let i = 1; i < appState.forensicSnapshots.length; i++) {
			const diff = Math.abs(appState.forensicSnapshots[i].timestamp - targetTs);
			if (diff < minDiff) {
				minDiff = diff;
				nearest = appState.forensicSnapshots[i];
			}
		}
		return nearest.engines;
	}
	return appState.engines;
}

// ── Logging ──
const MAX_LOGS = 200;

export function addLog(level: LogEntry['level'], message: string, source = 'SYSTEM') {
	appState.logs = [
		{ timestamp: Date.now(), level, message, source },
		...appState.logs.slice(0, MAX_LOGS - 1)
	];
}

// ── Forensic Snapshot Capture ──
const SNAPSHOT_INTERVAL_MS = 60000; // 1 minute
const MAX_SNAPSHOTS = 1440; // 24 hours at 1/min

export function captureForensicSnapshot(): void {
	appState.forensicSnapshots = [
		...appState.forensicSnapshots.slice(-(MAX_SNAPSHOTS - 1)),
		{
			timestamp: Date.now(),
			conviction: appState.conviction,
			engines: appState.engines.map((e) => ({ ...e }))
		}
	];
}

let snapshotTimer: ReturnType<typeof setInterval> | null = null;

export function startForensicCapture(): void {
	if (snapshotTimer) return;
	captureForensicSnapshot();
	snapshotTimer = setInterval(captureForensicSnapshot, SNAPSHOT_INTERVAL_MS);
}

export function stopForensicCapture(): void {
	if (snapshotTimer) {
		clearInterval(snapshotTimer);
		snapshotTimer = null;
	}
}

// ══════════════════════════════════════════════════════
//  API Client — HTTPS (Fetch)
// ══════════════════════════════════════════════════════

interface ApiOptions {
	method?: string;
	body?: unknown;
	headers?: Record<string, string>;
	timeout?: number;
}

async function apiFetch<T>(endpoint: string, options: ApiOptions = {}): Promise<T> {
	const { method = 'GET', body, headers = {}, timeout = 15000 } = options;

	const controller = new AbortController();
	const timeoutId = setTimeout(() => controller.abort(), timeout);

	try {
		const response = await fetch(`${API_BASE}${endpoint}`, {
			method,
			headers: {
				'Content-Type': 'application/json',
				...headers
			},
			body: body ? JSON.stringify(body) : undefined,
			signal: controller.signal
		});

		if (!response.ok) {
			const errorText = await response.text().catch(() => 'Unknown error');
			throw new Error(`API ${response.status}: ${errorText}`);
		}

		return (await response.json()) as T;
	} catch (err) {
		if (err instanceof DOMException && err.name === 'AbortError') {
			throw new Error(`API timeout after ${timeout}ms: ${endpoint}`);
		}
		throw err;
	} finally {
		clearTimeout(timeoutId);
	}
}

// ── Trading Commands ──

export async function startTrading(): Promise<void> {
	addLog('INFO', 'Enviando comando de ignición...', 'API');
	try {
		const result = await apiFetch<{ status: string }>('/engine/mode/consensus', {
			method: 'POST'
		});
		appState.systemStatus = 'RUNNING';
		addLog('INFO', `Trading iniciado: ${result.status}`, 'API');
	} catch (err) {
		appState.systemStatus = 'ERROR';
		addLog('ERROR', `Fallo de ignición: ${(err as Error).message}`, 'API');
		throw err;
	}
}

export async function stopTrading(): Promise<void> {
	addLog('INFO', 'Deteniendo sistema...', 'API');
	try {
		await apiFetch('/engine/mode/isolation', { 
			method: 'POST',
			body: { target_engine: 'gap_sniper' }
		});
		appState.systemStatus = 'IDLE';
		addLog('INFO', 'Sistema en modo aislamiento (detenido para consenso)', 'API');
	} catch (err) {
		addLog('ERROR', `Error al detener: ${(err as Error).message}`, 'API');
		throw err;
	}
}

let configSyncTimeout: ReturnType<typeof setTimeout> | null = null;

export function updateConfig(config: CalibrationConfig): void {
	// Optimistic UI update
	appState.calibration = { ...config };
	appState.calibrationSyncing = true;
	
	if (configSyncTimeout) clearTimeout(configSyncTimeout);
	
	configSyncTimeout = setTimeout(async () => {
		try {
			await apiFetch('/engine/config', {
				method: 'POST',
				body: config
			});
			addLog('INFO', 'Configuración calibrada exitosamente', 'API');
		} catch (err) {
			addLog('ERROR', `Error al calibrar: ${(err as Error).message}`, 'API');
		} finally {
			appState.calibrationSyncing = false;
		}
	}, 500); // 500ms debounce
}

export async function toggleSystem(): Promise<void> {
	const wasRunning = appState.systemStatus === 'RUNNING';

	if (wasRunning) {
		await stopTrading();
	} else {
		await startTrading();
	}
}

export async function runBacktest(params?: Record<string, unknown>): Promise<void> {
	addLog('INFO', 'Iniciando simulación interna...', 'BACKTEST');
	appState.simulationResults = { status: 'RUNNING' };
	try {
		// Calling the only available test endpoint for now
		const result = await apiFetch<SimulationResults>('/test-sniper', {
			method: 'GET',
		});
		appState.simulationResults = { ...result, status: 'COMPLETED' };
		addLog('INFO', `Simulación completada | Retorno: ${result.totalReturn}%`, 'BACKTEST');
	} catch (err) {
		appState.simulationResults = { status: 'FAILED' };
		addLog('ERROR', `Simulación fallida: ${(err as Error).message}`, 'BACKTEST');
		throw err;
	}
}

export async function fetchStatus(): Promise<void> {
	try {
		const result = await apiFetch<any>('/engine/status');
		appState.systemStatus = result.mode === 'consensus' ? 'RUNNING' : 'IDLE';
		addLog('INFO', `Estado del sistema: ${result.mode}`, 'POLLING');
	} catch (err) {
		addLog('WARN', `Polling fallido: ${(err as Error).message}`, 'POLLING');
	}
}

// ══════════════════════════════════════════════════════
//  WebSocket — Live Telemetry (WSS)
// ══════════════════════════════════════════════════════

let ws: WebSocket | null = null;
let reconnectAttempt = 0;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let pingInterval: ReturnType<typeof setInterval> | null = null;

const MAX_RECONNECT_ATTEMPTS = 10;
const BASE_DELAY_MS = 1000;
const MAX_DELAY_MS = 30000;

function getBackoffDelay(attempt: number): number {
	return Math.min(BASE_DELAY_MS * Math.pow(2, attempt), MAX_DELAY_MS);
}

export function connectWebSocket(): void {
	if (ws?.readyState === WebSocket.OPEN || ws?.readyState === WebSocket.CONNECTING) {
		return;
	}

	appState.connectionStatus = reconnectAttempt > 0 ? 'RECONNECTING' : 'CONNECTING';
	addLog('INFO', `Conectando WSS (intento ${reconnectAttempt + 1})...`, 'WSS');

	try {
		ws = new WebSocket(`${WSS_BASE}/ws/telemetry`);
	} catch {
		scheduleReconnect();
		return;
	}

	ws.onopen = () => {
		reconnectAttempt = 0;
		appState.connectionStatus = 'CONNECTED';
		appState.telemetry.connected = true;
		appState.telemetry.retryCount = 0;
		addLog('INFO', 'Conexión WSS establecida', 'WSS');

		// Start ping/pong heartbeat
		if (pingInterval) clearInterval(pingInterval);
		pingInterval = setInterval(() => {
			if (ws?.readyState === WebSocket.OPEN) {
				const pingTime = Date.now();
				ws.send(JSON.stringify({ type: 'ping', timestamp: pingTime }));
			}
		}, 15000);
	};

	ws.onmessage = (event) => {
		try {
			const data = JSON.parse(event.data as string);
			handleTelemetryMessage(data);
		} catch {
			addLog('WARN', 'Mensaje WSS no parseable', 'WSS');
		}
	};

	ws.onclose = (event) => {
		cleanup();
		if (!event.wasClean) {
			addLog('WARN', `WSS cerrado inesperadamente (code: ${event.code})`, 'WSS');
			scheduleReconnect();
		} else {
			appState.connectionStatus = 'DISCONNECTED';
			addLog('INFO', 'WSS desconectado limpiamente', 'WSS');
		}
	};

	ws.onerror = () => {
		addLog('ERROR', 'Error en conexión WSS', 'WSS');
	};
}

function handleTelemetryMessage(data: Record<string, unknown>): void {
	const type = data.type as string;

	switch (type) {
		case 'pong':
			appState.telemetry.lastPing = Date.now();
			appState.telemetry.latencyMs = Date.now() - (data.timestamp as number);
			break;

		case 'status':
			appState.systemStatus = data.status as SystemStatus;
			break;

		case 'signal':
			appState.currentSignals = [data.payload as Signal, ...appState.currentSignals.slice(0, 49)];
			break;

		case 'final_conviction':
			appState.conviction = Math.max(0, Math.min(100, data.value as number));
			break;

		case 'engine_update': {
			const engineId = data.engine_id as string;
			const idx = appState.engines.findIndex((e) => e.id === engineId);
			if (idx !== -1) {
				appState.engines[idx] = {
					...appState.engines[idx],
					status: (data.status as EngineState['status']) ?? appState.engines[idx].status,
					weight: (data.weight as number) ?? appState.engines[idx].weight,
					lastVerdict: (data.verdict as string) ?? appState.engines[idx].lastVerdict,
					lastUpdate: Date.now(),
					pulseActive: true
				};
				// Auto-clear pulse after 800ms
				setTimeout(() => {
					if (appState.engines[idx]) {
						appState.engines[idx].pulseActive = false;
					}
				}, 800);
			}
			break;
		}

		case 'log':
			addLog(
				(data.level as LogEntry['level']) || 'INFO',
				data.message as string,
				(data.source as string) || 'BACKEND'
			);
			break;

		default:
			break;
	}
}

function scheduleReconnect(): void {
	if (reconnectAttempt >= MAX_RECONNECT_ATTEMPTS) {
		appState.connectionStatus = 'DISCONNECTED';
		appState.systemStatus = 'ERROR';
		addLog('ERROR', `Reconexión fallida tras ${MAX_RECONNECT_ATTEMPTS} intentos`, 'WSS');
		return;
	}

	const delay = getBackoffDelay(reconnectAttempt);
	appState.connectionStatus = 'RECONNECTING';
	appState.telemetry.retryCount = reconnectAttempt + 1;
	addLog('INFO', `Reintentando en ${Math.round(delay / 1000)}s...`, 'WSS');

	reconnectTimer = setTimeout(() => {
		reconnectAttempt++;
		connectWebSocket();
	}, delay);
}

function cleanup(): void {
	appState.telemetry.connected = false;
	if (pingInterval) {
		clearInterval(pingInterval);
		pingInterval = null;
	}
}

export function disconnectWebSocket(): void {
	if (reconnectTimer) {
		clearTimeout(reconnectTimer);
		reconnectTimer = null;
	}
	reconnectAttempt = MAX_RECONNECT_ATTEMPTS; // Prevent auto-reconnect
	if (ws) {
		ws.close(1000, 'Client disconnect');
		ws = null;
	}
	cleanup();
	appState.connectionStatus = 'DISCONNECTED';
	addLog('INFO', 'WSS desconectado por usuario', 'WSS');
}

// ── Initialize on import ──
export function initializeApp(): void {
	addLog('INFO', '𝔑𝔘𝔏𝔏 inicializando...', 'SYSTEM');
	fetchStatus();
	connectWebSocket();
}
