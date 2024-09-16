/* Copyright 2024 Brian Marre
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/** @file CalculateIPDInput ionization potential depression(IPD) sub-stage
 *
 * implements calculation of IPD input parameters from the local sumField values
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stewartPyatt/LocalIPDInputFields.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stewartPyatt/SumFields.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stewartPyatt/kernel/CalculateIPDInput.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/LocalTimeRemainingField.hpp"

#include <string>

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::stewartPyatt::stage
{
    /** IPD sub-stage for calculating IPD input from sumFields, required for calculating IPD
     *
     * @tparam T_numberAtomicPhysicsIonSpecies specialization template parameter used to prevent compilation of all
     *  atomicPhysics kernels if no atomic physics species is present.
     */
    template<uint32_t T_numberAtomicPhysicsIonSpecies>
    struct CalculateIPDInput
    {
        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            auto& localTimeRemainingField = *dc.get<
                picongpu::particles::atomicPhysics::localHelperFields::LocalTimeRemainingField<picongpu::MappingDesc>>(
                "LocalTimeRemainingField");

            auto& localSumWeightAllField
                = *dc.get<stewartPyatt::localHelperFields::SumWeightAllField<picongpu::MappingDesc>>(
                    "SumWeightAllField");
            auto& localSumTemperatureFunctionalField
                = *dc.get<stewartPyatt::localHelperFields::SumTemperatureFunctionalField<picongpu::MappingDesc>>(
                    "SumTemperatureFunctionalField");

            auto& localSumWeightElectronField
                = *dc.get<stewartPyatt::localHelperFields::SumWeightElectronsField<picongpu::MappingDesc>>(
                    "SumWeightElectronsField");

            auto& localSumChargeNumberIonsField
                = *dc.get<stewartPyatt::localHelperFields::SumChargeNumberIonsField<picongpu::MappingDesc>>(
                    "SumChargeNumberIonsField");
            auto& localSumChargeNumberSquaredIonsField
                = *dc.get<stewartPyatt::localHelperFields::SumChargeNumberSquaredIonsField<picongpu::MappingDesc>>(
                    "SumChargeNumberSquaredIonsField");

            auto& localTemperatureEnergyField
                = *dc.get<stewartPyatt::localHelperFields::LocalTemperatureEnergyField<picongpu::MappingDesc>>(
                    "LocalTemperatureEnergyField");
            auto& localZStarField
                = *dc.get<stewartPyatt::localHelperFields::LocalZStarField<picongpu::MappingDesc>>("LocalZStarField");
            auto& localDebyeLengthField
                = *dc.get<stewartPyatt::localHelperFields::LocalDebyeLengthField<picongpu::MappingDesc>>(
                    "LocalDebyeLengthField");

            // macro for kernel call
            PMACC_LOCKSTEP_KERNEL(stewartPyatt::kernel::CalculateIPDInputKernel<T_numberAtomicPhysicsIonSpecies>())
                .template config<1u>(mapper.getGridDim())(
                    mapper,
                    localTimeRemainingField.getDeviceDataBox(),
                    localSumWeightAllField.getDeviceDataBox(),
                    localSumTemperatureFunctionalField.getDeviceDataBox(),
                    localSumWeightElectronField.getDeviceDataBox(),
                    localSumChargeNumberIonsField.getDeviceDataBox(),
                    localSumChargeNumberSquaredIonsField.getDeviceDataBox(),
                    localTemperatureEnergyField.getDeviceDataBox(),
                    localZStarField.getDeviceDataBox(),
                    localDebyeLengthField.getDeviceDataBox());
        }
    };

    template<>
    struct CalculateIPDInput<0u>
    {
        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
        }
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::stewartPyatt::stage
