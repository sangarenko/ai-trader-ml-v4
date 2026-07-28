/**
 * IStrategy - interface for all trading strategies.
 */
import { Candle } from '../core/types'
import { SniperEvolvedStrategy, EvolvedParams } from './sniper-evolved'

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
    default:
      throw new Error(`Unknown strategy: ${name}`)
  }
}
