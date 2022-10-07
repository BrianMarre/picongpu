/* Copyright 2022 Sergei Bastrakov, Brian Marre, Axel Huebl, Rene Widera
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

#pragma once

#include <cstdint>
#include <iostream>

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics2
        {
            namespace debug
            {
                /** debug only, print content and bins of histogram to console, serial and cpu build only
                 */
                template<typename T_Histogram>
                void printHistogramToConsole(T_Histogram& histogram)
                {
                    constexpr uint32_t numBins = T_Histogram::numberBins;

                    std::cout << "histogram: base=" << histogram.getBase();
                    std::cout << " numBins=" << T_Histogram::numberBins;
                    std::cout << " maxE=" << T_Histogram::maxEnergy << std::endl;

                    float_X centralEnergy;
                    float_X binWidth;

                    for(uint32_t i = 0u; i < numBins; i++)
                    {
                        // binIndex
                        std::cout << "\t "<< i;

                        // central bin energy [eV] and binWidth [eV]
                        centralEnergy = histogram.getBinEnergy(i);
                        binWidth = histogram.getBinWidth(i);

                        std::cout << "(" << centralEnergy - binWidth/2._X << ", "<<  centralEnergy + binWidth/2._X << "] :";

                        // bin data, [w0, DeltaW, DeltaEnergy, binOverSubscribed]
                        std::cout << " [w0, Dw, DE, o?]: [";
                        std::cout << histogram.getBinWeight0(i) << ", ";
                        std::cout << histogram.getBinDeltaWeight(i) << ", ";
                        std::cout << histogram.getBinDeltaEnergy(i) << ", ";
                        std::cout << histogram.isBinOverSubscribed(i) << "]";
                        std::cout << std::endl;
                    }
                    std::cout << "\t overFlow: w0=" << histogram.getOverflowWeight() << std::endl;
                }
            } // namespace debug
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu