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

/** @file implements the ionization potential depression(IPD) calculation according to the Barrier Suppression
 *  Ionization(BSI) model.
 */

#pragma once

#include "picongpu/defines.hpp"

#include <cstdint>

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::fieldBarrierSuppression
{
    struct Implementation_BSI:
    {
        /** calculate IPD due to the electric field
         *
         * @param eFieldNorm, in internal units
         * @param atomicStateCollectionIndex, index of the atomic state in the collection of atomic states
         * @param chargeStateDataBox dataBox giving access to the charge state property data
         * @param atomicStateDataBox dataBox giving access to the atomic state property data
         */
        template<typename T_ChargeStateDataBox, typename T_AtomicStateDataBox>
        HDINLINE static float_X calculateIPD(
            float_X const eFieldNorm,
            uint32_t const atomicStateCollectionIndex,
            T_ChargeStateDataBox const chargeStateDataBox,
            T_AtomicStateDataBox const atomicStateDataBox)
        {
            auto const configNumber = atomicStateDataBox.configNumber(atomicStateCollectionIndex);

            using S_ConfigNumber = typename T_AtomicStateDataBox::ConfigNumber;
            uint8_t const chargeState = S_ConfigNumber::getChargeState(configNumber);

            // e
            float_X const screenedCharge = chargeStateDataBox.screenedCharge(chargeState) - 1._X;

            return calculateIPD(eFieldNorm, screenedCharge);
        }
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::fieldBarrierSuppression
