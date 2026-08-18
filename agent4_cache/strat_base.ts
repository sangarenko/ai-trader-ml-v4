$ cat /opt/ai-trader/src/strategies/base.ts
--- rc=0 ---
/**
 * IStrategy - interface for all trading strategies.
 */
import { Candle } from '../core/types'
import { SniperEvolvedStrategy, EvolvedParams } from './sniper-evolved'
import { MultiTimeframeStrategy, MultiTimeframeParams } from './multi_timeframe'

export interface StrategyContext {
  ticker?: string
  entryPrice?: number
  holding?: number  // signed qty: negative = short, needed for take-profit side
}

export interface IStrategy {
  name: string
  description: string
  predict(candles: Candle[], idx: number, hasPosition: boolean, stepsHeld?: number, ctx?: StrategyContext): number
}

export function createStrategy(name: string, config: any): IStrategy {
  switch (name) {
    case 'sniper-v2': {
      const { SniperTrendV2Strategy } = require('./sniper-v2')
      return new SniperTrendV2Strategy()
    }
    case 'sniper-v5': {
      const { SniperTrendV5Strategy } = require('./sniper-v5')
      return new SniperTrendV5Strategy(config.paramsFile)
    }
    case 'sniper-v6': {
      const { SniperTrendV6Strategy } = require('./sniper-v6')
      return new SniperTrendV6Strategy()
    }
    case 'sniper-v7': {
      const { SniperTrendV7Strategy } = require('./sniper-v7')
      return new SniperTrendV7Strategy()
    }
    case 'sniper-v8a': {
      const { SniperTrendV8aStrategy } = require('./sniper-v8a')
      return new SniperTrendV8aStrategy()
    }
    case 'sniper-v8b': {
      const { SniperTrendV8bStrategy } = require('./sniper-v8b')
      return new SniperTrendV8bStrategy()
    }
    case 'sniper-v10': {
      const { SniperTrendV10Strategy } = require('./sniper-v10')
      return new SniperTrendV10Strategy()
    }
    case 'sniper-v11': {
      const { SniperTrendV11Strategy } = require('./sniper-v11')
      return new SniperTrendV11Strategy()
    }
    case 'sniper-v14': {
      return new SniperEvolvedStrategy(config.params as EvolvedParams)
    }
    case 'sniper-v15': {
      return new SniperEvolvedStrategy(config.params as EvolvedParams)
    }
    case 'sniper-v13': {
      return new SniperEvolvedStrategy(config.params as EvolvedParams)
    }
    case 'sniper-v12': {
      const { SniperTrendV12Strategy } = require('./sniper-v12')
      return new SniperTrendV12Strategy()
    }
    case 'sniper-v9': {
      const { SniperTrendV9Strategy } = require('./sniper-v9')
      return new SniperTrendV9Strategy()
    }

    case 'sniper-v17': {
      const { SniperTrendV17Strategy } = require('./sniper-v17')
      return new SniperTrendV17Strategy()
    }
    case 'sniper-v18': {
      const { SniperTrendV18Strategy } = require('./sniper-v18')
      return new SniperTrendV18Strategy()
    }
    case 'sniper-v19': {
      const { SniperTrendV19Strategy } = require('./sniper-v19')
      return new SniperTrendV19Strategy()
    }

    case 'rsi-revert': {
      const { RSIReversionV2Strategy } = require('./rsi-revert')
      return new RSIReversionV2Strategy()
    }
    case 'sma-crossover': {
      const { SMACrossStrategy } = require('./sma-crossover')
      return new SMACrossStrategy()
    }
    case 'momentum-simple': {
      const { MomentumSimpleStrategy } = require('./momentum-simple')
      return new MomentumSimpleStrategy()
    }
    case 'mathematician': {
      const { MathematicianStrategy } = require('./mathematician')
      return new MathematicianStrategy(config.modelFile)
    }
    case 'trained-dqn': {
      const { TrainedDQNStrategy } = require('./trained-dqn')
      return new TrainedDQNStrategy(config.modelFile, config.epsilon || 0.15)
    }
    case 'pairs-trading': {
      const { PairsTradingStrategy } = require('./pairs-trading')
      return new PairsTradingStrategy() as any
    }
    case 'aboba': {
      const { AbobaStrategy } = require('./aboba')
      return new AbobaStrategy(config.pretrainedFile)
    }
    case 'multi_timeframe': {
      return new MultiTimeframeStrategy(config.params as MultiTimeframeParams)
    }
    case 'wiseplat_triple_sma': {
      const { WiseplatTripleSmaStrategy } = require('./wiseplat_triple_sma')
      return new WiseplatTripleSmaStrategy()
    }
    case 'turtle_donchian': {
      const { TurtleDonchianStrategy } = require('./turtle_donchian')
      return new TurtleDonchianStrategy()
    }
    case 'rsi_extremes': {
      const { RsiExtremesStrategy } = require('./rsi_extremes')
      return new RsiExtremesStrategy()
    }
    case 'bollinger_bounce': {
      const { BollingerBounceStrategy } = require('./bollinger_bounce')
      return new BollingerBounceStrategy()
    }
    case 'macd_trend': {
      const { MacdTrendStrategy } = require('./macd_trend')
      return new MacdTrendStrategy()
    }
    case 'vwap_reversion': {
      const { VwapReversionStrategy } = require('./vwap_reversion')
      return new VwapReversionStrategy()
    }
    case 'momentum_volume': {
      const { MomentumVolumeStrategy } = require('./momentum_volume')
      return new MomentumVolumeStrategy()
    }
    case 'random_hold_short': {
      // MC01 Monte Carlo winner — uses multi_timeframe params
      return new MultiTimeframeStrategy(config.params as MultiTimeframeParams)
    }
    case 'connors_rsi2': {
      const { ConnorsRSI2Strategy } = require('./connors_rsi2')
      return new ConnorsRSI2Strategy()
    }
    case 'zscore_reversion': {
      const { ZScoreReversionStrategy } = require('./zscore_reversion')
      return new ZScoreReversionStrategy()
    }
    case 'supertrend': {
      const { SupertrendStrategy } = require('./supertrend')
      return new SupertrendStrategy()
    }
    case 'bollinger_squeeze': {
      const { BollingerSqueezeStrategy } = require('./bollinger_squeeze')
      return new BollingerSqueezeStrategy()
    }
    case 'atr_bands': {
      const { AtrBandsStrategy } = require('./atr_bands')
      return new AtrBandsStrategy()
    }
    case 'heikin_ashi': {
      const { HeikinAshiTrendStrategy } = require('./heikin_ashi')
      return new HeikinAshiTrendStrategy()
    }
    case 'dual_thrust': {
      const { DualThrustStrategy } = require('./dual_thrust')
      return new DualThrustStrategy()
    }
    case 'awesome_oscillator': {
      const { AwesomeOscillatorStrategy } = require('./awesome_oscillator')
      return new AwesomeOscillatorStrategy()
    }
    case 'golden_cross': {
      const { GoldenCrossStrategy } = require('./golden_cross')
      return new GoldenCrossStrategy()
    }
    case 'orb': {
      const { OpeningRangeBreakoutStrategy } = require('./orb')
      return new OpeningRangeBreakoutStrategy()
    }
    case 'stoch_oscillator': {
      const { StochOscillatorStrategy } = require('./stoch_oscillator')
      return new StochOscillatorStrategy()
    }
    case 'v2_short': {
      // ALIAS: v2_short → sniper-v2 (no .ts file provided, uses sniper-v2 logic)
      const { SniperTrendV2Strategy } = require('./sniper-v2')
      return new SniperTrendV2Strategy()
    }
    case 'v2_inverted': {
      // ALIAS: v2_inverted → sniper-v2 (no .ts file provided, uses sniper-v2 logic)
      const { SniperTrendV2Strategy } = require('./sniper-v2')
      return new SniperTrendV2Strategy()
    }
    case 'mean_reversion': {
      // ALIAS: mean_reversion → bollinger_bounce
      const { BollingerBounceStrategy } = require('./bollinger_bounce')
      return new BollingerBounceStrategy()
    }
    case 'trend_follow': {
      // ALIAS: trend_follow → macd_trend
      const { MacdTrendStrategy } = require('./macd_trend')
      return new MacdTrendStrategy()
    }
    case 'bb_reversion': {
      // ALIAS: bb_reversion → bollinger_bounce
      const { BollingerBounceStrategy } = require('./bollinger_bounce')
      return new BollingerBounceStrategy()
    }
    case 'donchian_breakout': {
      // ALIAS: donchian_breakout → turtle_donchian
      const { TurtleDonchianStrategy } = require('./turtle_donchian')
      return new TurtleDonchianStrategy()
    }
    case 'ml_predict': {
      const { MLPredictStrategy } = require('./ml_predict')
      return new MLPredictStrategy()
    }
    case 'ml_predict_v2': {
      const { MLPredictV2Strategy } = require('./ml_predict_v2')
      return new MLPredictV2Strategy()
    }
    case 'meta_selector': {
      const { MetaSelectorStrategy } = require('./meta_selector')
      return new MetaSelectorStrategy()
    }
    case 'meta_selector_v4': {
      const { MetaSelectorV4Strategy } = require('./meta_selector_v4')
      return new MetaSelectorV4Strategy()
    }
    default:
      throw new Error(`Unknown strategy: ${name}`)
  }
}


