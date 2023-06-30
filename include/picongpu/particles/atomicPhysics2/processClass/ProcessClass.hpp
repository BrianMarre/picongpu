/* Copyright 2023 Brian Marre
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software you can redistribute it andor modify
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

/** @file processClass enum
 *
 * @attention NumberIonizationelectrons and NumebrPhysicalTransitions must be kept consistent
 *  with the enum ProcessClass
 */

#pragma once

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics2
{
    namespace processClass
    {
        enum struct ProcessClass : uint8_t
        {
            noChange = 0u,
            spontaneousDeexcitation = 1u,
            electronicExcitation = 2u,
            electronicDeexcitation = 3u,
            electronicIonization = 4u,
            autonomousIonization = 5u,
            fieldIonization = 6u
        };
    } // namespace processClass

    template<processClass::ProcessClass T_ProcessClass>
    ALPAKA_FN_HOST std::string enumToString()
    {
        if constexpr(
            static_cast<uint8_t>(T_ProcessClass) == static_cast<uint8_t>(processClass::ProcessClass::noChange))
            return "noChange";
        if constexpr(
            static_cast<uint8_t>(T_ProcessClass)
            == static_cast<uint8_t>(processClass::ProcessClass::spontaneousDeexcitation))
            return "spontaneousDeexcitation";
        if constexpr(
            static_cast<uint8_t>(T_ProcessClass)
            == static_cast<uint8_t>(processClass::ProcessClass::electronicExcitation))
            return "electronicExcitation";
        if constexpr(
            static_cast<uint8_t>(T_ProcessClass)
            == static_cast<uint8_t>(processClass::ProcessClass::electronicDeexcitation))
            return "electronicDeexcitation";
        if constexpr(
            static_cast<uint8_t>(T_ProcessClass)
            == static_cast<uint8_t>(processClass::ProcessClass::electronicIonization))
            return "electronicIonization";
        if constexpr(
            static_cast<uint8_t>(T_ProcessClass)
            == static_cast<uint8_t>(processClass::ProcessClass::autonomousIonization))
            return "autonomousIonization";
        if constexpr(
            static_cast<uint8_t>(T_ProcessClass) == static_cast<uint8_t>(processClass::ProcessClass::fieldIonization))
            return "fieldIonization";
        return "unknown";
    }
} // namespace picongpu::particles::atomicPhysics2
