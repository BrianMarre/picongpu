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
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/fields/SimulationFieldHelper.hpp>
#include <pmacc/memory/boxes/PitchedBox.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>

#include <cstdint>
#include <string>

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
                    : public pmacc::ISimulationData
                    , public pmacc::SimulationFieldHelper<T_MappingDescription>
                {
                    //! Type of data box for field values on host and device
                    using DataBoxType = pmacc::DataBox<pmacc::PitchedBox<T_Histogram, simDim>>;

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

                    /// @todo should be private?
                    //! pointer to gridBuffer of histograms:T_histograms created upon creation
                    std::unique_ptr<pmacc::GridBuffer<T_Histogram, simDim>> localHistogramField;
                    /// @todo should these be private?
                    //! type of physical particle represented in histogram, usually "Electron" or "Photon"
                    std::string histogramType;

                    LocalHistogramField(T_MappingDescription const& mappingDesc, std::string const histogramType)
                        : SimulationFieldHelper<T_MappingDescription>(mappingDesc)
                        , histogramType(histogramType)
                    {
                        this->localHistogramField
                            = std::make_unique<GridBuffer<T_Histogram, simDim>>(mappingDesc.getGridSuperCells());
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
                        localHistogramField->deviceToHost();
                    }

                    // required by SimulationFieldHelper
                    void reset(uint32_t currentStep) override
                    {
                         /// @todo figure out why exactly this way
                         localHistogramField->getHostBuffer().reset(true);
                         localHistogramField->getDeviceBuffer().reset(false);
                    };

                    // required by SimulationHelperField
                    //! ==hostToDevice
                    void syncToDevice() override
                    {
                        localHistogramField->hostToDevice();
                    }

                    //! get dataBox on device for use in device kernels
                    DataBoxType getDeviceDataBox()
                    {
                        return this->localHistogramField->getDeviceBuffer().getDataBox();
                    }
                };
            } // namespace electronHistogram
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu