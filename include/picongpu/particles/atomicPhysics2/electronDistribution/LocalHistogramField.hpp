/* Copyright 2022 Brian Marre
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

/** @file implements the local electron histogram field for each superCell
 *
 */

#pragma once

#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/fields/SimulationHelperField.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/dimensions/DataSpace.hpp>


#include <string>
#include <cstdint>

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics2
        {
            namespace electronDistribution
            {
                /**@class holds a gridBuffer of the per-superCell localHistograms for atomicPhysics
                 *
                 */
                template<typename T_Histogram, typename T_MappingDescription>
                struct LocalHistogramField
                    : public pmacc::ISimulationData,
                      public pmacc::SimulationHelperField<T_MappingDescription>
                {
                    /// @todo should these be private?
                    /* from SimulationFieldHelper<T_MappingDescription>:
                     * protected:
                     *      T_MappingDescription cellDescription;
                     * public:
                     *      static constexpr uint32_t dim = T_MappingDescription::dim;
                     *      using MappingDesc = T_MappingDescription;
                     *      void ~<> = default;
                     *
                     *      T_MappingDescription getCellDescription() const
                     */

                    //! pointer to gridBuffer of histograms:T_histograms created upon creation
                    std::unique_ptr<pmacc::GridBuffer<T_Histogram, simDim>> localHistogramField;
                    //! type of physical particle represented in histogram, usually "Electron" or "Photon"
                    std::string histogramType;

                    LocalHistogramField(T_MappingDescription const& mappingDesc, std::string const histogramType)
                        : SimulationHelperField<T_MappingDescription>(mappingDesc),
                          histogramType(histogramType)
                    {
                        this->localHistogramField = std::make_unique<GridBuffer<T_Histogram, simDim>>( mappingDesc.getGridSuperCells()::toRT() );
                    }

                    // required by ISimulationData
                    std::string getUniqueId() override
                    {
                        return histogramType + "_localHistogramField";
                    }

                    // required by ISimulationData
                    //! == deviceToHost
                    void synchronize() override
                    {
                        (this->localHistogramField)->deviceToHost();
                    }

                    // required by SimulationHelperField
                    void reset() override
                    {
                         /// @todo figure out why exactly this way
                        (this->localHistogramField)->getHostBuffer.reset(true);
                        (this->localHistogramField)->getDeviceBuffer.reset(false);
                    };

                    // required by SimulationHelperField
                    void syncToDevice() override
                    {
                        (this->localHistogramField)->hostToDevice()
                    }

                }
            } // namespace electronHistogram
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu