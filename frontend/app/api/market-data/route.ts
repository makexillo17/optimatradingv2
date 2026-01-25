import { NextResponse } from 'next/server';
import ccxt from 'ccxt';
import { EMA } from 'technicalindicators';

// Force dynamic to prevent caching at build time
export const dynamic = 'force-dynamic';

interface Candle {
    timestamp: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

export async function GET() {
    try {
        // 1. Fetch Data from Kraken
        const exchange = new ccxt.kraken({
            timeout: 10000,
        });

        // Fetch enough candles for EMA200
        const ohlcv = await exchange.fetchOHLCV('BTC/USD', '1h', undefined, 250);

        if (!ohlcv || ohlcv.length === 0) {
            throw new Error('No data received from Kraken');
        }

        // Parse data
        const closes = ohlcv.map(c => c[4]);
        const timestamps = ohlcv.map(c => c[0]);

        // 2. Calculate EMAs
        const ema50Values = EMA.calculate({ period: 50, values: closes });
        const ema200Values = EMA.calculate({ period: 200, values: closes });

        // Align data (EMA arrays are shorter)
        // We want to return the last N candles with their corresponding EMAs
        const limit = 100;
        const resultData = [];

        // Technical Indicators returns array starting from index (period-1).
        // e.g. for EMA50, index 0 corresponds to close[49].
        // So ema50Values[i] corresponds to closes[i + 49].

        const total = ohlcv.length;
        const startIndex = total - limit;

        for (let i = startIndex; i < total; i++) {
            if (i < 0) continue;

            // Calculate matching index for EMAs
            // Index in closes is 'i'.
            // Index in ema50 is 'i - (50-1)'.
            const idx50 = i - 49;
            const idx200 = i - 199;

            const val50 = idx50 >= 0 ? ema50Values[idx50] : null;
            const val200 = idx200 >= 0 ? ema200Values[idx200] : null;

            // Format timestamp for frontend
            const date = new Date(timestamps[i]);
            // Format: HH:mm
            const timeStr = date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });

            resultData.push({
                time: timeStr,
                timestamp: timestamps[i],
                open: ohlcv[i][1],
                high: ohlcv[i][2],
                low: ohlcv[i][3],
                close: ohlcv[i][4],
                ema50: val50,
                ema200: val200,
            });
        }

        // 3. Construct Response
        const currentPrice = closes[closes.length - 1];
        const prevPrice = closes[closes.length - 2];
        const change = ((currentPrice - prevPrice) / prevPrice) * 100;

        return NextResponse.json({
            symbol: 'BTC/USD',
            price: currentPrice,
            change: change,
            data: resultData
        });

    } catch (error: any) {
        console.error('API Error:', error);
        return NextResponse.json({ error: error.message }, { status: 500 });
    }
}
