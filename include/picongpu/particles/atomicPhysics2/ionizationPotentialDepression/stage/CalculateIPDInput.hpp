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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
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

#include "pciongpu/particles/AtomicPhysics2/localHelperFields/localTimeRemainingField.hpp"
#include "picongpu/particles/atomicPhysics2/ionizationPotentialDepression/kernel/CalculateIPDInput.kernel"

#include <string>

namespace picongpu::particles::atomicPhysics2::ionizationPotentialDepression::stage
{
    //! short hand for IPD namespace
    namespace s_IPD = picongpu::particles::atomicPhysics2::ionizationPotentialDepression;

    //! IPD sub-stage for calculating IPD input from sumFields, required for calculating IPD
    struct CalculateIPDInput
    {
        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
            pmacc::lockstep::WorkerCfg workerCfg = pmacc::lockstep::makeWorkerCfg<1u>();

            auto& localTimeRemainingField
                = *dc.get<picongpu::particles::atomicPhysics2::localHelperFields::LocalTimeRemainingField<
                    picongpu::MappingDesc>>("LocalTimeRemainingField");

            auto& localSumWeightAllField = *dc.get<s_IPD::localHelperFields::SumWeightAllField>("SumWeightAllField");
            auto& localSumTemperatureFunctionalField
                = *dc.get<s_IPD::localHelperFields::SumTemperatureFunctionalField>("SumTemperatureFunctionalField");

            auto& localSumWeightElectronField
                = *dc.get<s_IPD::localHelperFields::SumWeigthElectronsField>("SumWeightElectronsField");

            auto& localSumChargeNumberIonsField
                = *dc.get<s_IPD::localHelperFields::SumChargeNumberIonsField>("SumChargeNumberIonsField");
            auto& localSumChargeNumberSquaredIonsField
                = *dc.get<s_IPD::localHelperFields::SumChargeNumberSquaredIonsField>(
                    "SumChargeNumberSquaredIonsField");

            auto& localTemperatureEnergyField
                = *dc.get<s_IPD::localHelperFields::LocalTemperatureEnergyField>("LocalTemperatureEnergyField");
            auto& localZStarField = *dc.get<s_IPD::localHelperFields::LocalZStarField>("LocalZStarField");
            auto& localDebyeLengthField
                = *dc.get<s_IPD::localHelperFields::LocalDebyeLengthField>("LocalDebyeLengthField");

            // macro for kernel call
            PMACC_LOCKSTEP_KERNEL(s_IPD::kernel::CalculateIPDInputKernel(), workerCfg)
            (mapper.getGridDim())(
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
} // namespace picongpu::particles::atomicPhysics2::ionizationPotentialDepression::stage
