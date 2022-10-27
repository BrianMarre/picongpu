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

#include "picongpu/particles/atomicPhysics2/atomicData/AtomicTuples.def"
#include "picongpu/particles/atomicPhysics2/atomicData/TransitionDataBox.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>

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
                class BoundBoundTransitionDataBox :
                    public TransitionDataBox<
                        T_DataBoxType,
                        T_Number,
                        T_Value,
                        T_ConfigNumberDataType,
                        T_atomicNumber>
                {
                public:
                    using S_BoundBoundTransitionTuple = BoundBoundTransitionTuple<TypeValue, Idx>;

                private:
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
                     * @attention transition data must be sorted block-wise ascending by lower/upper
                     *  atomic state and secondary ascending by upper/lower atomic state.
                     *
                     * @param boxCollisionalOscillatorStrength
                     * @param boxAbsorptionOscillatorStrength
                     * @param boxCinx1 gaunt tunneling fit parameter 1
                     * @param boxCinx2 gaunt tunneling fit parameter 2
                     * @param boxCinx3 gaunt tunneling fit parameter 3
                     * @param boxCinx4 gaunt tunneling fit parameter 4
                     * @param boxCinx5 gaunt tunneling fit parameter 5
                     * @param boxLowerConfigNumber configNumber of the lower(lower excitation energy) state of the
                     * transition
                     * @param boxUpperConfigNumber configNumber of the upper(higher excitation energy) state of the
                     * transition
                     * @param numberTransitions number of atomic bound-bound transitions stored
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
                        BoxConfigNumber boxUpperConfigNumber,
                        uint32_t numberTransitions)
                        : m_boxCollisionalOscillatorStrength(boxCollisionalOscillatorStrength)
                        , m_boxAbsorptionOscillatorStrength(boxAbsorptionOscillatorStrength)
                        , m_boxCinx1(boxCinx1)
                        , m_boxCinx2(boxCinx2)
                        , m_boxCinx3(boxCinx3)
                        , m_boxCinx4(boxCinx4)
                        , m_boxCinx5(boxCinx5)
                        , TransitionDataBox(boxLowerConfigNumber, boxUpperConfigNumber, numberTransitions)
                    {
                    }

                    /** store transition in data box
                     *
                     * @attention do not forget to call syncToDevice() on the
                     *  corresponding buffer, or the state is only added on the host side.
                     * @attention needs to fulfill all ordering and content assumptions of constructor!
                     * @attention no range checks outside debug compile!
                     *
                     * @param collectionIndex index of data box entry to rewrite
                     * @param tuple tuple containing data of transition
                     */
                    HINLINE void store(uint32_t const collectionIndex, S_BoundBoundTransitionTuple& tuple)
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex >= m_numberTransitions)
                        {
                            std::runtime_error("atomicPhysics ERROR: out of range store()");
                            return;
                        }
                        m_boxCollisionalOscillatorStrength[collectionIndex] = std::get<0>(tuple);
                        m_boxAbsorptionOscillatorStrength[collectionIndex] = std::get<1>(tuple);
                        m_boxCinx1[collectionIndex] = std::get<2>(tuple);
                        m_boxCinx2[collectionIndex] = std::get<3>(tuple);
                        m_boxCinx3[collectionIndex] = std::get<4>(tuple);
                        m_boxCinx4[collectionIndex] = std::get<5>(tuple);
                        m_boxCinx5[collectionIndex] = std::get<6>(tuple);
                        storeTransitions(collectionIndex, std::get<7>(tuple), std::get<8>(tuple));
                    }

                    /** returns collisional oscillator strength of the transition
                     *
                     * @param collectionIndex ... collection index of transition
                     *
                     * @attention no range checks outside debug compile
                     */
                    HDINLINE TypeValue getCollisionalOscillatorStrength(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex >= m_numberTransitions)
                        {
                            printf("atomicPhysics ERROR: outside range call getCollisionalOscillatorStrength\n");
                            return static_cast<ValueType>(0._X);
                        }
                        return m_boxCollisionalOscillatorStrength(indexTransition);
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
                        if(collectionIndex >= m_numberTransitions)
                        {
                            printf("atomicPhysics ERROR: outside range call getAbsorptionOscillatorStrength\n");
                            return static_cast<ValueType>(0._X);
                        }
                        return m_boxAbsorptionOscillatorStrength(indexTransition);
                    }

                    /// @todo find way to replace cinx getters with single template function

                    /** returns gaunt tunneling fit parameter 1 of the transition
                     *
                     * @param collectionIndex ... collection index of transition
                     *
                     * @attention no range checks outside debug compile
                     */
                    HDINLINE TypeValue getCinx1(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex >= m_numberTransitions)
                        {
                            printf("atomicPhysics ERROR: outside range call getCinx1\n");
                            return static_cast<ValueType>(0._X);
                        }
                        return m_boxCinx1(indexTransition);
                    }

                    /** returns gaunt tunneling fit parameter 2 of the transition
                     *
                     * @param collectionIndex ... collection index of transition
                     *
                     * @attention no range checks outside debug compile
                     */
                    HDINLINE TypeValue getCinx2(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex >= m_numberTransitions)
                        {
                            printf("atomicPhysics ERROR: outside range call getCinx2\n");
                            return static_cast<ValueType>(0._X);
                        }
                        return m_boxCinx2(indexTransition);
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
                        if(collectionIndex >= m_numberTransitions)
                        {
                            printf("atomicPhysics ERROR: outside range call getCinx3\n");
                            return static_cast<ValueType>(0._X);
                        }
                        return m_boxCinx3(indexTransition);
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
                        if(collectionIndex >= m_numberTransitions)
                        {
                            printf("atomicPhysics ERROR: outside range call getCinx4\n");
                            return static_cast<ValueType>(0._X);
                        }
                        return m_boxCinx4(indexTransition);
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
                        if(collectionIndex >= m_numberTransitions)
                        {
                            printf("atomicPhysics ERROR: outside range call getCinx5\n");
                            return static_cast<ValueType>(0._X);
                        }
                        return m_boxCinx5(indexTransition);
                    }

                };

                /** complementing buffer class
                 *
                 * @tparam T_Number dataType used for number storage, typically uint32_t
                 * @tparam T_Value dataType used for value storage, typically float_X
                 * @tparam T_atomicNumber atomic number of element this data corresponds to, eg. Cu -> 29
                 */
                template<
                    typename T_DataBoxType,
                    typename T_Number,
                    typename T_Value,
                    typename T_ConfigNumberDataType,
                    uint8_t T_atomicNumber>
                class BoundBoundTransitionDataBuffer : public TransitionDataBuffer< T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber>
                {
                    std::unique_ptr< BufferValue > bufferCollisionalOscillatorStrength;
                    std::unique_ptr< BufferValue > bufferAbsorptionOscillatorStrength;
                    std::unique_ptr< BufferValue > bufferCinx1;
                    std::unique_ptr< BufferValue > bufferCinx2;
                    std::unique_ptr< BufferValue > bufferCinx3;
                    std::unique_ptr< BufferValue > bufferCinx4;
                    std::unique_ptr< BufferValue > bufferCinx5;

                public:
                    /** buffer corresponding to the above dataBox object
                     *
                     * @param numberAtomicStates number of atomic states, and number of buffer entries
                     */
                    HINLINE BoundBoundTransitionDataBuffer(uint32_t numberBoundBoundTransitions)
                        : TransitionDataBuffer(numberBoundBoundTransitions)
                    {

                        auto const guardSize = pmacc::DataSpace<1>::create(0);
                        auto const layoutBoundboundTransitions = pmacc::GridLayout<1>(numberBoundBoundTransitions, guardSize);

                        bufferCollisionalOscillatorStrength.reset( new BufferValue(layoutBoundBoundTransitions));
                        bufferAbsorptionOscillatorStrength.reset( new BufferValue(layoutBoundBoundTransitions));
                        bufferCinx1.reset( new BufferValue(layoutBoundBoundTransitions));
                        bufferCinx2.reset( new BufferValue(layoutBoundBoundTransitions));
                        bufferCinx3.reset( new BufferValue(layoutBoundBoundTransitions));
                        bufferCinx4.reset( new BufferValue(layoutBoundBoundTransitions));
                        bufferCinx5.reset( new BufferValue(layoutBoundBoundTransitions));
                    }

                    HINLINE BoundBoundTransitionDataBox< T_DataBoxType, T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber> getHostDataBox()
                    {
                        return BoundBoundTransitionDataBox< T_DataBoxType, T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber>(
                            bufferCollisionalOscillatorStrength->getHostBuffer().getDataBox(),
                            bufferAbsorptionOscillatorStrength->getHostBuffer().getDataBox(),
                            bufferCinx1->getHostBuffer().getDataBox(),
                            bufferCinx2->getHostBuffer().getDataBox(),
                            bufferCinx3->getHostBuffer().getDataBox(),
                            bufferCinx4->getHostBuffer().getDataBox(),
                            bufferCinx5->getHostBuffer().getDataBox(),
                            bufferLowerConfigNumber->getHostBuffer().getDataBox(),
                            bufferUpperConfigNumber->getHostBuffer().getDataBox(),
                            m_numberTransitions);
                    }

                    HINLINE BoundBoundTransitionDataBox< T_DataBoxType, T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber> getDeviceDataBox()
                    {
                        return BoundBoundTransitionDataBox< T_DataBoxType, T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber>(
                            bufferCollisionalOscillatorStrength->getDeviceBuffer().getDataBox(),
                            bufferAbsorptionOscillatorStrength->getDeviceBuffer().getDataBox(),
                            bufferCinx1->getDeviceBuffer().getDataBox(),
                            bufferCinx2->getDeviceBuffer().getDataBox(),
                            bufferCinx3->getDeviceBuffer().getDataBox(),
                            bufferCinx4->getDeviceBuffer().getDataBox(),
                            bufferCinx5->getDeviceBuffer().getDataBox(),
                            bufferLowerConfigNumber->getDeviceBuffer().getDataBox(),
                            bufferUpperConfigNumber->getDeviceBuffer().getDataBox(),
                            m_numberTransitions);
                    }

                    HINLINE void syncToDevice()
                    {
                        bufferCollisionalOscillatorStrength->hostToDevice();
                        bufferAbsorptionOscillatorStrength->hostToDevice();
                        bufferCinx1->hostToDevice();
                        bufferCinx2->hostToDevice();
                        bufferCinx3->hostToDevice();
                        bufferCinx4->hostToDevice();
                        bufferCinx5->hostToDevice();
                        syncToDevice_BaseClass();
                    }

                };

            } // namespace atomicData
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
