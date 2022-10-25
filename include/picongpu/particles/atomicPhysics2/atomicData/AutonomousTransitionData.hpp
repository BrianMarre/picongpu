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

#include "picongpu/particles/atomicPhysics2/atomicData/TransitionDataBox.hpp"

#include <cstdint>

/** @file implements the storage of autonomous transitions property data
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
                /** data box storing bound-free transition property data
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
                    uint8_t T_atomicNumber>
                class AutonomousTransitionDataBox :
                    public TransitionDataBox<
                        T_DataBoxType,
                        T_Number,
                        T_Value,
                        T_ConfigNumberDataType,
                        T_atomicNumber>
                {
                    /// @todo better unit?, Brian Marre, 2022
                    BoxValue m_boxTransitionRate; // unit: 1/s

                    /** constructor
                     *
                     * @attention transition data must be sorted block-wise by atomic state
                     *  and secondary ascending by upper configNumber.
                     *
                     * @param boxTransitionRate rate over deexcitation [1/s]
                     * @param boxLowerConfigNumber configNumber of the lower(lower excitation energy) state of the transition
                     * @param boxUpperConfigNumber configNumber of the upper(higher excitation energy) state of the transition
                     * @param numberTransitions number of atomic bound-bound transitions stored
                     * @param numberTransitions number of atomic autonomous transitions stored
                     */

                    AutonomousTransitionDataBox(
                        BoxValue boxTransitionRate,
                        BoxConfigNumber boxLowerConfigNumber,
                        BoxConfigNumber boxUpperConfigNumber,
                        uint32_t numberTransitions)
                        : m_boxTransitionRate(boxTransitionRate)
                        , TransitionDataBox(boxLowerConfigNumber, boxUpperConfigNumber, numberTransitions)
                    {
                    }

                   /** returns rate of the transition
                    *
                    * @param collectionIndex ... collection index of transition
                    *
                    * @attention no range checks
                    */
                    TypeValue getTransitionRate(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex < T_numberTransitions)
                            return m_boxTransitionRate(indexTransition);
                        return static_cast<ValueType>(0._X);
                    }

                };

                /** complementing buffer class
                 *
                 * @tparam T_Number dataType used for number storage, typically uint32_t
                 * @tparam T_Value dataType used for value storage, typically float_X
                 * @tparam T_atomicNumber atomic number of element this data corresponds to, eg. Cu -> 29
                 */
                template<
                    typename T_DataBoxType
                    typename T_Number,
                    typename T_Value,
                    typename T_ConfigNumberDataType,
                    uint8_t T_atomicNumber>
                class AutonomousTransitionDataBuffer : public TransitionDataBuffer< T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber>
                {
                    std::unique_ptr< BufferValue > bufferTransitionRate;

                public:
                    /** buffer corresponding to the above dataBox object
                     *
                     * @param numberAtomicStates number of atomic states, and number of buffer entries
                     */
                    HINLINE AutonomousTransitionDataBuffer(uint32_t numberAutonomousTransitions)
                        : TransitionDataBuffer(numberBoundBoundTransitions)
                    {

                        auto const guardSize = pmacc::DataSpace<1>::create(0);
                        auto const layoutAutonomousTransitions = pmacc::GridLayout<1>(numberBoundBoundTransitions, guardSize);

                        bufferTransitionRate.reset( new BufferValue(layoutAutonomousTransitions));
                    }

                    HINLINE BoundBoundTransitionDataBox< T_DataBoxType, T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber> getHostDataBox()
                    {
                        return BoundBoundTransitionDataBox< T_DataBoxType, T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber>(
                            bufferTransitionRate->getHostBuffer().getDataBox(),
                            bufferLowerConfigNumber->getHostBuffer().getDataBox(),
                            bufferUpperConfigNumber->getHostBuffer().getDataBox(),
                            m_numberTransitions);
                    }

                    HINLINE BoundBoundTransitionDataBox< T_DataBoxType, T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber> getDeviceDataBox()
                    {
                        return BoundBoundTransitionDataBox< T_DataBoxType, T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber>(
                            bufferTransitionRate->getDeviceBuffer().getDataBox(),
                            bufferLowerConfigNumber->getDeviceBuffer().getDataBox(),
                            bufferUpperConfigNumber->getDeviceBuffer().getDataBox(),
                            m_numberTransitions);
                    }

                    HINLINE void syncToDevice()
                    {
                        bufferTransitionRate->hostToDevice();
                        bufferLowerConfigNumber->hostToDevice();
                        bufferUpperConfigNumber->hostToDevice();
                    }

                };

            } // namespace atomicData
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
