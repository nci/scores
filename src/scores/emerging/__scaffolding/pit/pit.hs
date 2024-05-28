#!/usr/bin/env cabal
{- cabal:
build-depends:
  base,
-}

import GHC.Float
import GHC.Float.RealFracMethods
import Debug.Trace

-- extents of a real space containing the domain (min, max)
-- e.g. for determining support of a function
newtype Extent = Extent (Float, Float) deriving (Show, Eq)

makeExtent :: Float -> Float -> Extent
makeExtent x y
  | x <= y = Extent (x, y)
  | otherwise = Extent (y, x)

-- parameters used for the underlying beta-distribution
-- https://en.wikipedia.org/wiki/Beta_distribution
data BetaParams = BetaParams
  { _betaA :: Rational,
    _betaB :: Rational
  }

gammaHalf :: Float
gammaHalf = 1.772

-- only takes values 1/2, 1, or positive integers
gammaFunc :: Rational -> Float
gammaFunc n
  | n == (1 / 2) = gammaHalf
  | n == 1 = 1
  | n >= 2 =
      int2Float $
        product [1 .. float2Int (fromRational n - 1)]
  | otherwise = error "Unsupported input"

-- probability distribution function for a beta distribution
-- Note: this is a crude approximation for testing only
betaPdf :: BetaParams -> Float -> Float
betaPdf (BetaParams _a _b) x = (n1 * n2) * d3 / (d1 * d2)
  where
    _f1 = map gammaFunc [_a, _b, _a + _b]
    _f2 =
      zipWith
        (\n' p' -> n' `powerFloat` fromRat p')
        [x, 1 - x]
        [_a - 1, _b - 1]
    [d1, d2, d3] = trace ("gammaOut: [d1, d2, d3] = " ++ show _f1) _f1
    [n1, n2] = trace ("betaFunc: [n1, n2] = " ++ show _f2) _f2

main :: IO ()
main = do
  print $ betaPdf (BetaParams (1/2) (1/2)) 0.01
