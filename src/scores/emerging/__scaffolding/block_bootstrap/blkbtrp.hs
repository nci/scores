#!/usr/bin/env cabal
{- cabal:
build-depends:
  base,
  transformers,
  vector,
  random
-}
{-# LANGUAGE TupleSections #-}

import Control.Monad.Trans.State.Strict
import qualified Data.Vector as V
import System.Random
import System.Random.Stateful

data AxisInfo = AxisInfo
  { _axisLen :: Int,
    _axisBlockSize :: Int,
    _axisNumBlocks :: Int
  }
  deriving (Eq, Show)

type AxisBlockIdx = (V.Vector (V.Vector Int))
type AxisBlockIdxExpanded = (V.Vector (V.Vector Int))

-- | Struct to hold axis information. Trims axis if block size is not a multiple of axis length.
mkAxisInfo :: Int -> Int -> AxisInfo
mkAxisInfo l b =
  case l `divMod` b of
    (0, 0) -> error "Empty block"
    (n, 0) -> AxisInfo l b n
    (n, _r) -> AxisInfo (n * b) b n -- trim

-- | Make cyclic block indices based on blocksize
mkCyclicBlockIdx :: AxisInfo -> Int -> V.Vector Int
mkCyclicBlockIdx (AxisInfo l b _) = V.unfoldrExactN b (\n -> (n `mod` l, n + 1))

-- | Indices for one block sample for an axis
mkAxisBlockSampleM :: AxisInfo -> State StdGen (V.Vector Int)
mkAxisBlockSampleM a@(AxisInfo l _ _) =
  mkCyclicBlockIdx a <$> state (randomR (0, l - 1))

-- | Indices for N block samples for an axis
mkAxisBlockSampleNM :: AxisInfo -> State StdGen AxisBlockIdx
mkAxisBlockSampleNM a@(AxisInfo _ _ n) =
  V.replicateM n (mkAxisBlockSampleM a)

-- | (Ordered) list containing n * b block sample indices for each axis,
-- where n = num blocks
--       b = block size
--       return = list containing n * b array of indices
btstrpIndices :: [AxisInfo] -> State StdGen [AxisBlockIdx]
btstrpIndices = mapM mkAxisBlockSampleNM

-- | Unfolds indices for axes based on axis length
indexRange :: [AxisInfo] -> AxisBlockIdx
indexRange = foldMap (\x -> V.singleton (V.iterateN (_axisLen x) (+ 1) 0))

-- | Expand indices into a vector of coordinates. A coordinate is a Vector of integers representing
-- the outermost -> innermost axis
expandIndices :: AxisBlockIdx -> AxisBlockIdxExpanded
expandIndices = foldr (\z acc -> foldMap (\x -> V.cons x <$> acc) z) (V.singleton V.empty)

main :: IO ()
main = do
  let axi = mkAxisInfo 10 5
  let axi2 = mkAxisInfo 21 7
  let axi3 = mkAxisInfo 8 2
  let pureGen = mkStdGen 32
  let idxRange = indexRange [axi, axi3]
  print axi
  print idxRange
  print $ expandIndices idxRange
  print $ evalState (btstrpIndices [axi, axi2]) pureGen
