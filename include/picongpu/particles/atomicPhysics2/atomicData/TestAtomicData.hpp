/* Copyright 2022-2023 Brian Marre
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

//! @file atomicData implementation for fixed rate matrix tests

#pragma once

#include "picongpu/simulation_defines.hpp"

#include <pmacc/dataManagement/ISimulationData.hpp>

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhyscis2::atomicData
{
    struct TestAtomicData : public pmacc::ISimulationData
    {
        using S_ChargeStateDataBox = ;
        using S_AtomicStateDataBox = ;
        using S_AutonomousTransitionDataBox = ;

     private:
        const std::string m_speciesName;

     public:
        TestAtomicData(
            std::string ,
            std::string ,
            std::string ,
            std::string ,
            std::string ,
            std::string speciesName)
            : m_speciesName(speciesName) = default;

        void hostToDevice() = default;

        void deviceToHost() = default;

        template<bool hostData>
        S_ChargeStateDataBox getChargeStateDataDataBox()

        template<bool hostData>
        S_AtomicStateDataBox getAtomicStateDataDataBox()

        template<bool hostData>
        S_AutonomousTransitionDataBox getAutonomousTransitionDataBox()

        template<bool hostData>
        S_AutonomousTransitionDataBox getInverseAutonomousTransitionDataBox()

        void synchronize() override
        {
            this->deviceToHost();
        }

        //! required by ISimulationData
        std::string getUniqueId() override
        {
            return m_speciesName + "_atomicData";
        }
    };
} // namespace picongpu::particles::atomicPhyscis2::atomicData
