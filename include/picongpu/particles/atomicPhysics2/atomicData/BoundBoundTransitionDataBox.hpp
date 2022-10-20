/* Copyright 2020-2022 Sergei Bastrakov, Brian Marre
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

#include "picongpu/particles/atomicPhysics2/atomicData/TransitionDataBox.hpp"

#include <cstdint>

/** @file implements the storage of bound-bound transitions property data
 *
 * @attention ConfigNumber specifies the number of a state as defined by the configNumber
 *      class, while index always refers to a collection index.
 *      The configNumber of a given state is always the same, its collection index depends
 *      on input file,it should therefore only be used internally!
 */

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics2
        {
            namespace atomicData
            {
                /** data box storing bound-bound transition property data
                 *
                 * for use on device.
                 *
                 * @tparam T_DataBoxType dataBox type used for storage
                 * @tparam T_Number dataType used for number storage, typically uint32_t
                 * @tparam T_Value dataType used for value storage, typically float_X
                 * @tparam T_ConfigNumberDataType dataType used for configNumber storage,
                 *      typically uint64_t
                 * @tparam T_TransitionIndexDataType dataType used for transition index,
                 *      typically uint32_t
                 * @tparam T_atomicNumber atomic number of element this data corresponds to, eg. Cu -> 29
                 * @tparam T_numberTransitions number of atomic bound-bound transitions stored
                 *
                 * @attention ConfigNumber specifies the number of a state as defined by the configNumber
                 *      class, while index always refers to a collection index.
                 *      The configNumber of a given state is always the same, its collection index depends
                 *      on input file,it should therefore only be used internally!
                 */
                template<
                    typename T_DataBoxType,
                    typename T_Number,
                    typename T_Value,
                    typename T_ConfigNumberDataType,
                    uint8_t T_atomicNumber,
                    uint32_t T_numberTransitions>
                class BoundBoundTransitionDataBox :
                    public TransitionDataBox<
                        T_DataBoxType,
                        T_Number,
                        T_Value,
                        T_ConfigNumberDataType,
                        T_atomicNumber>
                {
                    //! unitless
                    BoxValue m_boxCollisionalOscillatorStrength;
                    //! unitless
                    BoxValue m_boxAbsorptionOscillatorStrength;
                    //! gaunt tunneling fit parameter 1, unitless
                    BoxValue m_boxCinx1;
                    //! gaunt tunneling fit parameter 2, unitless
                    BoxValue m_boxCinx2;
                    //! gaunt tunneling fit parameter 3, unitless
                    BoxValue m_boxCinx3;
                    //! gaunt tunneling fit parameter 4, unitless
                    BoxValue m_boxCinx4;
                    //! gaunt tunneling fit parameter 5, unitless
                    BoxValue m_boxCinx5;

                public:
                    /** constructor
                     *
                     * @attention transition data must be sorted block-wise by atomic state
                     *  and secondary ascending by upper configNumber.
                     *
                     * @param boxCollisionalOscillatorStrength
                     * @param boxAbsorptionOscillatorStrength
                     * @param boxCinx1 gaunt tunneling fit parameter 1
                     * @param boxCinx2 gaunt tunneling fit parameter 2
                     * @param boxCinx3 gaunt tunneling fit parameter 3
                     * @param boxCinx4 gaunt tunneling fit parameter 4
                     * @param boxCinx5 gaunt tunneling fit parameter 5
                     * @param boxLowerConfigNumber configNumber of the lower(lower excitation energy) state of the transition
                     * @param boxUpperConfigNumber configNumber of the upper(higher excitation energy) state of the transition
                     */
                    BoundBoundTransitionDataBox(
                        BoxValue boxCollisionalOscillatorStrength,
                        BoxValue boxAbsorptionOscillatorStrength,
                        BoxValue boxCinx1,
                        BoxValue boxCinx2,
                        BoxValue boxCinx3,
                        BoxValue boxCinx4,
                        BoxValue boxCinx5,
                        BoxConfigNumber boxLowerConfigNumber,
                        BoxConfigNumber boxUpperConfigNumber)
                        : m_boxCollisionalOscillatorStrength(boxCollisionalOscillatorStrength)
                        , m_boxAbsorptionOscillatorStrength(boxAbsorptionOscillatorStrength)
                        , m_boxCinx1(boxCinx1)
                        , m_boxCinx2(boxCinx2)
                        , m_boxCinx3(boxCinx3)
                        , m_boxCinx4(boxCinx4)
                        , m_boxCinx5(boxCinx5)
                        , TransitionDataBox(boxLowerConfigNumber, boxUpperConfigNumber)
                        {
                        }

                   /** returns collisional oscillator strength of the transition
                    *
                    * @param collectionIndex ... collection index of transition
                    *
                    * @attention no range checks
                    */
                    HDINLINE TypeValue getCollisionalOscillatorStrength(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex < T_numberTransitions)
                            return m_boxCollisionalOscillatorStrength(indexTransition);
                        return static_cast<ValueType>(0._X);
                    }

                   /** returns absorption oscillator strength of the transition
                    *
                    * @param collectionIndex ... collection index of transition
                    *
                    * @attention no range checks
                    */
                    HDINLINE TypeValue getAbsorptionOscillatorStrength(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex < T_numberTransitions)
                            return m_boxAbsorptionOscillatorStrength(indexTransition);
                        return static_cast<ValueType>(0._X);
                    }

                    /// @todo find way to replace cinx getters with single template function

                   /** returns gaunt tunneling fit parameter 1 of the transition
                    *
                    * @param collectionIndex ... collection index of transition
                    *
                    * @attention no range checks
                    */
                    HDINLINE TypeValue getCinx1(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex < T_numberTransitions)
                            return m_boxCinx1(indexTransition);
                        return static_cast<ValueType>(0._X);
                    }

                   /** returns gaunt tunneling fit parameter 2 of the transition
                    *
                    * @param collectionIndex ... collection index of transition
                    *
                    * @attention no range checks
                    */
                    HDINLINE TypeValue getCinx2(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex < T_numberTransitions)
                            return m_boxCinx2(indexTransition);
                        return static_cast<ValueType>(0._X);
                    }

                   /** returns gaunt tunneling fit parameter 3 of the transition
                    *
                    * @param collectionIndex ... collection index of transition
                    *
                    * @attention no range checks
                    */
                    HDINLINE TypeValue getCinx3(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex < T_numberTransitions)
                            return m_boxCinx3(indexTransition);
                        return static_cast<ValueType>(0._X);
                    }

                   /** returns gaunt tunneling fit parameter 4 of the transition
                    *
                    * @param collectionIndex ... collection index of transition
                    *
                    * @attention no range checks
                    */
                    HDINLINE TypeValue getCinx4(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex < T_numberTransitions)
                            return m_boxCinx4(indexTransition);
                        return static_cast<ValueType>(0._X);
                    }

                   /** returns gaunt tunneling fit parameter 5 of the transition
                    *
                    * @param collectionIndex ... collection index of transition
                    *
                    * @attention no range checks
                    */
                    HDINLINE TypeValue getCinx5(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex < T_numberTransitions)
                            return m_boxCinx5(indexTransition);
                        return static_cast<ValueType>(0._X);
                    }


                };

            } // namespace atomicData
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
