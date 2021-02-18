/* Copyright 2020-2021 Brian Marre
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/** @file This file implements a adaptive Histogram starting from argument 0
 *
 * basic idea: 1-dimensional infinite histogram, with sparsely populated bins
 * binWidths are not uniform, but indirectly definded by a target value of a
 * relative binning width dependent error.
 *
 * only populated bins are stored in memory.
 *
 * TemplateParameters:
 * -------------------
 *  @tparam T_RelativeError ... functor with operator() which returns relative error
 *  @tparam T_AtomicDataBox ... type of data container for atomic input data
 *  @tparam T_maxNumberBins ... maximum number of bins of the histogram
 *  @tparam T_mayNumNewBins ... maximum number of new bins before updateWithNewBins
 *                                must be called by one thread
 *
 * private members:
 * ----------------
 *  float_X[T_maxNumberBins] binWeights ... histogram bin values
 *                                          weights of particles in bin in supercell
 *  float_X{T_maxNumberBins] binDeltaEnergy  ... secondary histogram value
 *                                               energy change in this time step in this histogram bin
 *  float_X{T_maxNumberBins] binLeftBoundary ... left boundary of bin in argument space
 *
 * storage of bins added since last call to updateWithNewBins
 * float_X[T_maxNumNewBins] newBinsWeights      ... see regular
 * float_X[T_maxNumNewBins] newBinsLeftBoundary ... -||-
 *  NOTE: no <newBinsDeltaEnergy> necessary, since all electrons are already binned for this step
 *
 *  float_x relativeErrorTarget   ... relative error target used to choose bin size
 *  T_RelativeError relativeError ... relative error functor
 *  float_X initialgridWidth      ... intial binWidth used
 *
 *  uint numBins    ... number of bins occupied
 *  uint numNewBins ... number of newBins occupied,
 *
 *  float_X lastLeftBoundary ... last left boundary of a bin used
 *
 * private methods:
 * ---------------
 *  uint findBin(float_X leftBoundary, uint startIndex=0) ... tries to finds leftBoundary in bin collection
 *  bool hasBin(float_X leftBoundary, uint startIndex=0)  ... checks wether bin exists
 *  float_X centerBin(bool directionPositive, float_X Boundary, float_X binWidth ) ...
 *      return center of Bin,
 *      direction positive indicates wether the leftBoundary(true) or the right Boundary(false) is given as argument
 *  bool inBin(bool directionPositive, float_X boundary, float_X binWidth, float_X x) ...
 *      is x in the given Bin?
 *
 * public methods:
 * ---------------
 *  void init(float_X relativeErrorTarget >0 , float_X initialGridWidth >0 , T_RelativeError& relativeError) ...
 *      init method for adaptive histogram, must be called by one thread once before use
 *      BEWARE: assumptions for input parameters
 *
 *  uint getMaxNumberBins()     ... returns corresponding template parameter value
 *  uint getMaxNUmberNewBins()  ... -||-
 *  float_X getInitialGridWidth()
 *
 *  uint getNumBins() ... returns current number of occupied bins
 *  float_X getEnergyBin(T_Acc& acc, uint index, T_AtomicDataBox atomicDataBox) ...
 *      returns central energy of bin with given collection index if it exists, otherwise returns 0._X
 *  float_X getLeftBoundaryBin(uint index) ... return leftBoundary of bin with given collection index
 *  float_X getWeightBin(uint index) ... returns weight of Bin with given collection index
 *      BEWARE: does not check if the bin is occupied, or exists at all,
 *          leave alone unless you know what you are doing
 *  float_X getDeltaEnergyBin(uint index) ... same as above for delta energy bin, beware also applies
 *  uint getBinIndex(T_Acc& acc, float_X energy, T_AtomicDataBox atomicDataBox) ...
 *      returns collection index of existing bin containing this energy, or maxNumBins otherwise
 *  float_X getBinWidth(T_Acc& acc, bool directionPositive, float_X boundary, float_X currentBinWidth,
 *      T_AtomicDataBox atomicDataBox) ...
 *      return binWidth of the specifed bin, for more see in code function documentation
 */

#pragma once

#include "picongpu/simulation_defines.hpp"
#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <alpaka/alpaka.hpp>

#include <utility>
#include "picongpu/traits/attribute/GetMass.hpp"


namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics
        {
            namespace electronDistribution
            {
                namespace histogram2
                {
                    template<
                        uint32_t T_maxNumBins,
                        uint32_t T_maxNumNewBins,
                        typename T_RelativeError,
                        typename T_AtomicDataBox>
                    struct AdaptiveHistogram
                    {
                    private:
                        //{ members
                        constexpr static uint32_t maxNumBins = T_maxNumBins;
                        constexpr static uint32_t maxNumNewBins = T_maxNumNewBins;

                        // content of bins
                        // two data fields
                        // 1. weight of particles
                        float_X binWeights[maxNumBins];
                        // 2. change in particle Energy in this bin
                        float_X binDeltaEnergy[maxNumBins];

                        // location of bins
                        float_X binLeftBoundary[maxNumBins];

                        // number of bins occupied, <= T_maxNumBins
                        uint16_t numBins;

                        // new bins since last call to update method
                        float_X newBinsLeftBoundary[maxNumNewBins];
                        float_X newBinsWeights[maxNumNewBins];
                        // number of entries in new bins
                        uint32_t numNewBins;

                        // reference point of histogram, always boundary of a bin in the histogram
                        // in current implementation currently not used
                        float_X lastLeftBoundary; // unit: Argument

                        // target value of relative Error of a histogram bin,
                        // bin width choosen such that relativeError is close to target
                        float_X relativeErrorTarget; // unit: varies
                        // relative error functor
                        T_RelativeError relativeError;

                        // defines initial global grid
                    public:
                        float_X initialGridWidth; // unit: Argument
                    private:
                        //}

                        // TODO: replace linear search, by ordering bins
                        /** Tries to find binLeftBoundary in the collection,
                         * starting at the given collection index and return the collection index.
                         *
                         * Returns index in the binIndices array when present or maxNumBin when not present
                         *
                         * maxNumBin is never a valid index of the collection
                         */
                        DINLINE uint16_t findBin(
                            float_X leftBoundary, // unit: Argument
                            uint16_t startIndex = 0u) const
                        {
                            for(uint16_t i = startIndex; i < numBins; i++)
                            {
                                if(this->binLeftBoundary[i] == leftBoundary)
                                {
                                    return i;
                                }
                            }
                            // case bin not found
                            return maxNumBins;
                        }

                        // checks whether Bin exists
                        DINLINE bool hasBin(float_X leftBoundary) const
                        {
                            auto const index = findBin(leftBoundary);
                            return (index < maxNumBins);
                        }

                        // return center of Bin
                        DINLINE static float_X centerBin(
                            bool const directionPositive,
                            float_X const Boundary,
                            float_X const binWidth)
                        {
                            if(directionPositive)
                                return Boundary + binWidth / 2._X;
                            else
                                return Boundary - binWidth / 2._X;
                        }


                        /** =^= is x in Bin?
                         *  case directionPositive == true:    [boundary, boundary + binWidth)
                         *  case directionPositive == false:   [boundary - binWidth,boundary)
                         */
                        DINLINE static bool inBin(
                            bool directionPositive,
                            float_X boundary,
                            float_X binWidth,
                            float_X x)
                        {
                            if(directionPositive)
                                return (x >= boundary) && (x < boundary + binWidth);
                            else
                                return (x >= boundary - binWidth) && (x < boundary);
                        }

                    public:
                        /** Has to be called by one thread once before any other method
                         * of this object to correctly initialise the histogram.
                         *
                         * @param relativeErrorTarget ... should be >0,
                         *      maximum relative Error of Histogram Bin
                         * @param initialGridWidth ... should be >0,
                         *      starting binwidth of algorithm
                         * @param relativeError ... relative should be monoton rising with rising binWidth
                         */
                        DINLINE void init(
                            float_X relativeErrorTarget,
                            float_X initialGridWidth,
                            T_RelativeError& relativeError)
                        {
                            printf("     initialGridWidth_INIT %d\n", initialGridWidth);
                            // init histogram empty
                            this->numBins = 0u;
                            this->numNewBins = 0u;

                            // init with 0 as reference point
                            this->lastLeftBoundary = 0._X;

                            // init adaptive bin width algorithm parameters
                            this->relativeErrorTarget = relativeErrorTarget;
                            this->initialGridWidth = initialGridWidth;
                            printf("     this->initialGridWidth_INIT %f\n", this->initialGridWidth);

                            // functor of relative error
                            this->relativeError = relativeError;

                            // TODO: make this debug mode only
                            // since we are filling memory we are never touching (if everything works)

                            //{ start of debug init
                            for(uint16_t i = 0u; i < maxNumBins; i++)
                            {
                                this->binWeights[i] = 0._X;
                                this->binDeltaEnergy[i] = 0._X;
                                this->binLeftBoundary[i] = 0._X;
                            }

                            for(uint16_t i = 0u; i < maxNumNewBins; i++)
                            {
                                this->newBinsLeftBoundary[i] = 0._X;
                                this->newBinsWeights[i] = 0._X;
                            }
                            //} end of debug init
                        }


                        DINLINE static uint16_t getMaxNumberBins()
                        {
                            return AdaptiveHistogram::maxNumBins;
                        }

                        DINLINE static constexpr uint16_t getMaxNumberNewBins()
                        {
                            return AdaptiveHistogram::maxNumNewBins;
                        }

                        DINLINE uint16_t getNumBins()
                        {
                            return this->numBins;
                        }


                        DINLINE float_X getInitialGridWidth()
                        {
                            return this->initialGridWidth;
                        }

                        // return unit: value
                        /** if bin with given index exists returns it's central energy, otherwise returns 0
                         *
                         * 0 is never a valid central energy since 0 is always a left boundary of a bin
                         */
                        template<typename T_Acc>
                        DINLINE float_X getEnergyBin(T_Acc& acc, uint16_t index, T_AtomicDataBox atomicDataBox) const
                        {
                            // no need to check for < 0, since uint
                            if(index < this->numBins)
                            {
                                float_X leftBoundary = this->binLeftBoundary[index];
                                float_X binWidth = this->getBinWidth(
                                    acc,
                                    true,
                                    leftBoundary,
                                    this->initialGridWidth,
                                    atomicDataBox,
                                    false // debug only
                                );

                                return leftBoundary + binWidth / 2._X;
                            }
                            // index outside range
                            return 0._X;
                        }

                        DINLINE float_X getLeftBoundaryBin(uint16_t index) const
                        {
                            // no need to check for < 0, since uint
                            if(index < this->numBins)
                            {
                                return this->binLeftBoundary[index];
                            }
                            // index outside range
                            return -1._X;
                        }


                        DINLINE float_X getWeightBin(uint16_t index) const
                        {
                            return this->binWeights[index];
                        }


                        DINLINE float_X getDeltaEnergyBin(uint16_t index) const
                        {
                            return this->binDeltaEnergy[index];
                        }

                        /** returns collection index of existing bin containing this energy
                         * or maxNumBins otherwise
                         */
                        template<typename T_Acc>
                        DINLINE uint16_t getBinIndex(
                            T_Acc& acc,
                            float_X energy, // unit: argument
                            T_AtomicDataBox atomicDataBox) const
                        {
                            float_X leftBoundary;
                            float_X binWidth;

                            for(uint16_t i = 0; i < this->numBins; i++)
                            {
                                leftBoundary = this->binLeftBoundary[i];
                                binWidth = this->getBinWidth(
                                    acc,
                                    true,
                                    leftBoundary,
                                    this->initialGridWidth,
                                    atomicDataBox,
                                    false // Debug only
                                );
                                if(this->inBin(true, leftBoundary, binWidth, energy))
                                {
                                    return i;
                                }
                            }
                            return this->maxNumBins;
                        }

                        /** find z, /in Z, iteratively, such that currentBinWidth * 2^z
                         * gives a lower relativeError than the relativeErrorTarget and maximises the
                         * binWidth.
                         *
                         * @param directionPositive .. whether the bin faces in positive argument
                         *      direction from the given boundary
                         * @param boundary
                         * @param currentBinWidth ... starting binWidth, the result may be both larger and
                         *      smaller than initial value
                         */
                        template<typename T_Acc>
                        DINLINE float_X getBinWidth(
                            T_Acc& acc,
                            const bool directionPositive,
                            const float_X boundary, // unit: value
                            float_X currentBinWidth,
                            T_AtomicDataBox atomicDataBox,
                            bool debug // debug only
                        ) const
                        {
                            // first check
                            // return 0.1_X;

                            // preparation for debug access to run time acess
                            uint32_t const workerIdx = cupla::threadIdx(acc).x;

                            // debug acess
                            if((workerIdx == 0) && debug)
                            {
                                // debug code
                                printf(
                                    "    +/-?: %s, initBinWidth: %f, boundary: %f\n",
                                    directionPositive ? "t" : "f",
                                    currentBinWidth,
                                    boundary);
                            }

                            // is initial binWidth realtiveError below the Target?
                            bool isBelowTarget
                                = (this->relativeErrorTarget >= this->relativeError(
                                       acc,
                                       currentBinWidth,
                                       AdaptiveHistogram::centerBin(directionPositive, boundary, currentBinWidth),
                                       atomicDataBox));

                            // debug only
                            uint16_t loopCounter = 0u;
                            uint16_t maxLoopCount = 0u;

                            if(isBelowTarget)
                            {
                                // increase until no longer below
                                while(isBelowTarget && (loopCounter <= maxLoopCount))
                                {
                                    // debug only
                                    loopCounter++;
                                    // try higher binWidth
                                    currentBinWidth *= 2._X;

                                    // until no longer below Target
                                    isBelowTarget
                                        = (this->relativeErrorTarget > this->relativeError(
                                               acc,
                                               currentBinWidth,
                                               AdaptiveHistogram::centerBin(
                                                   directionPositive,
                                                   boundary,
                                                   currentBinWidth),
                                               atomicDataBox));
                                }

                                // last i-th try was not below target,
                                // but (i-1)-th was still below
                                // -> reset to value i-1
                                currentBinWidth /= 2._X;
                            }
                            else
                            {
                                // decrease until below target for the first time
                                while((!isBelowTarget) && (loopCounter <= maxLoopCount))
                                {
                                    // debug only
                                    loopCounter++;
                                    // lower binWidth
                                    currentBinWidth /= 2._X;

                                    // until first time below Target
                                    isBelowTarget
                                        = (this->relativeErrorTarget >= this->relativeError(
                                               acc,
                                               currentBinWidth,
                                               AdaptiveHistogram::centerBin(
                                                   directionPositive,
                                                   boundary,
                                                   currentBinWidth),
                                               atomicDataBox));
                                }
                                // no need to reset to value before
                                // since this was first value that was below target
                            }
                            // debug acess
                            if((workerIdx == 0) && debug)
                            {
                                // debug code
                                printf(
                                    "    Final: +/-?: %s, initialBinWidth: %f, boundary: %f, loopN: %i\n",
                                    directionPositive ? "t" : "f",
                                    currentBinWidth,
                                    boundary,
                                    loopCounter);
                            }

                            return currentBinWidth;
                        }


                        /** Returns the left boundary of the bin a given argument value x belongs into
                         *
                         * see file description for more information
                         *
                         * @param x ... where object is located that we want to bin
                         * does not change lastBinLeftBoundary
                         */
                        template<typename T_Acc>
                        DINLINE float_X getBinLeftBoundary(
                            T_Acc& acc,
                            float_X const x,
                            T_AtomicDataBox atomicDataBox,
                            bool debug // debug only
                        ) const
                        {
                            // wether x is in positive direction with regards to last known
                            // Boundary
                            bool directionPositive = (x >= this->lastLeftBoundary);

                            // always use initial grid Width as starting point iteration
                            float_X currentBinWidth = this->initialGridWidth;

                            // start from prevoius point to reduce seek times
                            //  BEWARE: currently starting point does not change
                            // TODO: seperate this to seeker class with each worker getting it's own
                            //  instance
                            float_X boundary = this->lastLeftBoundary; // unit: argument

                            // preparation for debug access to run time acess
                            uint32_t const workerIdx = cupla::threadIdx(acc).x;
                            // debug acess
                            if(workerIdx == 0)
                            {
                                // debug code
                                printf("    initialBinWidth: %f, boundary: %f\n", currentBinWidth, boundary);
                            }

                            bool inBin = false;
                            while(!inBin)
                            {
                                // get currentBinWidth
                                // currentBinWidth = 0.1_X;
                                // debug sinc ethis call seems to cause infinite loop
                                currentBinWidth = getBinWidth(
                                    acc,
                                    directionPositive,
                                    boundary,
                                    currentBinWidth,
                                    atomicDataBox,
                                    false);

                                inBin = AdaptiveHistogram::inBin(directionPositive, boundary, currentBinWidth, x);

                                // check wether x is in Bin
                                if(inBin)
                                {
                                    if(directionPositive)
                                        // already left boundary of bin
                                        return boundary;
                                    else
                                        // right boundary of Bin
                                        return boundary - currentBinWidth;
                                }
                                else
                                {
                                    if(directionPositive)
                                    {
                                        boundary += currentBinWidth;
                                    }
                                    else
                                    {
                                        boundary -= currentBinWidth;
                                    }
                                }
                            }
                            return boundary;
                        }


                        template<typename T_Acc>
                        DINLINE void removeEnergyFromBin(
                            T_Acc& acc,
                            uint16_t index,
                            float_X deltaEnergy // unit: argument
                        )
                        {
                            cupla::atomicAdd(acc, &(this->binDeltaEnergy[index]), deltaEnergy);
                            // TODO: think about moving corresponding weight of electrons to lower energy bin
                            //      might be possible if bisn are without gaps, should make histogram filling
                            //      even easier and faster
                        }


                        // add for Argument x, weight to the bin
                        template<typename T_Acc>
                        DINLINE void binObject(
                            T_Acc const& acc,
                            float_X const x,
                            float_X const weight,
                            T_AtomicDataBox atomicDataBox)
                        {
                            // compute global bin index
                            float_X const binLeftBoundary = this->getBinLeftBoundary(
                                acc,
                                x,
                                atomicDataBox,
                                true // debug switch
                            );

                            // search for bin in collection of existing bins
                            auto const index = findBin(binLeftBoundary);

                            // If the bin was already there, we need to atomically increase
                            // the value, as another thread may contribute to the same bin
                            if(index < maxNumBins) // bin already exists
                            {
                                cupla::atomicAdd(acc, &(this->binWeights[index]), weight);
                            }
                            else
                            {
                                // Otherwise we add it to a collection of new bins
                                // Note: in current dev the namespace is different in cupla
                                // get Index where to deposit it by atomic add to numNewBins
                                // this assures that the same index is not used twice
                                auto newBinIdx
                                    = cupla::atomicAdd<alpaka::hierarchy::Threads>(acc, &numNewBins, uint32_t(1u));
                                if(newBinIdx < maxNumNewBins)
                                {
                                    newBinsWeights[newBinIdx] = weight;
                                    newBinsLeftBoundary[newBinIdx] = binLeftBoundary;
                                }
                                else
                                {
                                    /// If we are here, there were more new bins since the last update
                                    // call than memory allocated for them
                                    // Normally, this should not happen
                                }
                            }
                        }


                        /** This method moves new bins to the main collection of bins
                         * Should be called periodically so that we don't run out of memory for
                         * new bins
                         * choose maxNumNewBins to fit period and number threads and particle frame
                         * size
                         * MUST be called sequentially
                         */
                        DINLINE void updateWithNewBins()
                        {
                            uint32_t const numBinsBeforeUpdate = numBins;
                            for(uint32_t i = 0u; i < this->numNewBins; i++)
                            {
                                // New bins were definitely not present before
                                // But several new bins can be the same actual bin
                                // So we search in the newly added part only
                                auto const index = findBin(this->newBinsLeftBoundary[i], numBinsBeforeUpdate);

                                // If this bin was already added
                                if(index < maxNumBins)
                                    this->binWeights[index] += this->newBinsWeights[i];
                                else
                                {
                                    if(this->numBins < this->maxNumBins)
                                    {
                                        this->binWeights[this->numBins] = this->newBinsWeights[i];
                                        this->binLeftBoundary[this->numBins] = this->newBinsLeftBoundary[i];
                                        this->numBins++;
                                    }
                                    // else we ran out of memory, do nothing
                                }
                            }
                            this->numNewBins = 0u;
                        }
                    };

                } // namespace histogram2
            } // namespace electronDistribution
        } // namespace atomicPhysics
    } // namespace particles
} // namespace picongpu
