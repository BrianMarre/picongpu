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

/** @file implements the local timeRemainingField for each superCell
 *
 * timeRemaining for the current atomicPhysics step in each superCell
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
            /**@class holds a gridBuffer of the per-superCell timeRemaining:float_X for atomicPhysics
             *
             */
            template<typename T_MappingDescription>
            struct LocalTimeRemainingField
                : public pmacc::ISimulationData
                , public pmacc::SimulationFieldHelper<T_MappingDescription>
            {
                //! type of data box for field values on host and device
                using DataBoxType = pmacc::DataBox<pmacc::PitchedBox<float_X, simDim>>;
                //! type of device buffer
                using DeviceBufferType = pmacc::DeviceBuffer<float_X, simDim>;

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
                std::unique_ptr<pmacc::GridBuffer<float_X, simDim>> localTimeRemainingField;

                LocalTimeRemainingField(T_MappingDescription const& mappingDesc)
                    : SimulationFieldHelper<T_MappingDescription>(mappingDesc)
                {
                    this->localTimeRemainingField
                        = std::make_unique<GridBuffer<float_X, simDim>>(mappingDesc.getGridSuperCells());
                }

                // required by ISimulationData
                std::string getUniqueId() override
                {
                    return "LocalTimeRemaining";
                }

                // required by ISimulationData
                //! == deviceToHost
                void synchronize() override
                {
                    localTimeRemainingField->deviceToHost();
                }

                // required by SimulationFieldHelper
                void reset(uint32_t currentStep) override
                {
                    /// @todo figure out why exactly this way
                    localTimeRemainingField->getHostBuffer().reset(true);
                    localTimeRemainingField->getDeviceBuffer().reset(false);
                };

                // required by SimulationHelperField
                //! ==hostToDevice
                void syncToDevice() override
                {
                    localTimeRemainingField->hostToDevice();
                }

                /** get dataBox on device for use in device kernels
                 *
                 * Note: dataBoxes are just "pointers"
                 */
                DataBoxType getDeviceDataBox()
                {
                    return this->localTimeRemainingField->getDeviceBuffer().getDataBox();
                }

                DeviceBufferType& getDeviceBuffer()
                {
                    return this->localTimeRemainingField->getDeviceBuffer();
                }
            };
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu