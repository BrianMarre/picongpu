/* Copyright 2022 Sergei Bastrakov, Brian Marre
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

#include "picongpu/particles/atomicPhysics2/atomicData/Data.hpp"

#include <cstdint>
//#include <memory>
//#include <utility>

/** @file implements the storage of charge state property data
 *
 * The charge state data consists of the following data sets:
 *
 * - list of ionization states (sorted in ascending order of ionization):
 *      [ (ionization energy, [eV]
 *         screenedCharge, [eV] )]
 */

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics2
        {
            namespace atomicData
            {

                /** data box storing charge state property data
                 *
                 * for use on device.
                 *
                 * @tparam T_DataBoxType dataBox type used for storage
                 * @tparam T_Number dataType used for number storage, typically uint32_t
                 * @tparam T_Value dataType used for value storage, typically float_X
                 * @tparam T_atomicNumber atomic number of element this data corresponds to, eg. Cu -> 29
                 */
                template<
                    typename T_DataBoxType,
                    typename T_Number,
                    typename T_Value,
                    uint8_t T_atomicNumber // element
                    >
                class ChargeStateDataBox : Data<T_DataBoxType, T_Number, T_Value, T_atomicNumber>
                {
                    //! unit: eV
                    BoxValue m_boxIonizationEnergy;
                    //! unit: elementary charge
                    BoxValue m_boxScreenedCharge;

                public:
                    /** constructor
                     *
                     * @attention charge state data must be sorted by ascending charge and
                     * @attention the completely ionized state must be left out.
                     *
                     * @param ionizationEnergy ionization energy[eV] of charge states
                     * @param screenedCharge screenedCharge[e] of charge states
                     */
                    ChargeStateDataBox(
                        BoxValue ionizationEnergy,
                        BoxValue screenedCharge)
                        : m_boxIonizationEnergy(ionizationEnergy)
                        , m_boxScreenedCharge(screenedCharge)
                    {
                    }

                    //!@attention NEVER call with chargeState == T_atomicNumber, otherwise invalid memory access
                    T_Value ionizationEnergy(uint8_t chargeState)
                    {
                        return m_boxIonizationEnergy[chargeState];
                    }

                    //! @attention NEVER call with chargeState == T_atomicNumber, otherwise invalid memory access
                    T_Value screenedCharge(uint8_t chargeState)
                    {
                        return m_boxScreenedCharge[chargeState]
                    }
                };

            } // namespace atomicData
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu