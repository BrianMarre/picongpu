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

/** @file implements a superCell based field */

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
            /**@class holds a gridBuffer of one entry:T_Entry per per-superCell
             *
             * @tparam T_Entry type of the entry to store for each superCell
             * @tparam T_MappingDescription type used for description of mapping of
             *      simulation domain to memory
             */
            template<typename T_Entry, typename T_MappingDescription>
            struct SuperCellField
                : public pmacc::ISimulationData
                , public pmacc::SimulationFieldHelper<T_MappingDescription>
            {
                //! type of data box for field values on host and device
                using DataBoxType = pmacc::DataBox<pmacc::PitchedBox<T_Entry, simDim>>;
                //! type of device buffer
                using DeviceBufferType = pmacc::DeviceBuffer<T_Entry, simDim>;

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
                std::unique_ptr<pmacc::GridBuffer<T_Entry, simDim>> superCellField;

                SuperCellField(T_MappingDescription const& mappingDesc)
                    : SimulationFieldHelper<T_MappingDescription>(mappingDesc)
                {
                    this->superCellField
                        = std::make_unique<GridBuffer<T_Entry, simDim>>(mappingDesc.getGridSuperCells());
                }

                // required by ISimulationData
                virtual std::string getUniqueId() = 0;

                // required by ISimulationData
                //! == deviceToHost
                void synchronize() override
                {
                    superCellField->deviceToHost();
                }

                // required by SimulationFieldHelper
                void reset(uint32_t currentStep) override
                {
                    /// @todo figure out why exactly this way
                    superCellField->getHostBuffer().reset(true);
                    superCellField->getDeviceBuffer().reset(false);
                };

                // required by SimulationHelperField
                //! ==hostToDevice
                void syncToDevice() override
                {
                    superCellField->hostToDevice();
                }

                /** get dataBox on device for use in device kernels
                 *
                 * Note: dataBoxes are just "pointers"
                 */
                DataBoxType getDeviceDataBox()
                {
                    return this->superCellField->getDeviceBuffer().getDataBox();
                }

                DeviceBufferType& getDeviceBuffer()
                {
                    return this->superCellField->getDeviceBuffer();
                }
            };
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
