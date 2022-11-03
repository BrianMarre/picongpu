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
#include "picongpu/particles/atomicPhysics2/atomicData/TransitionData.hpp"

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
                class BoundFreeTransitionDataBox :
                    public TransitionDataBox<
                        T_DataBoxType,
                        T_Number,
                        T_Value,
                        T_ConfigNumberDataType,
                        T_atomicNumber>
                {
                public:
                    using S_BoundFreeTransitionTuple = BoundFreeTransitionTuple<TypeValue, Idx>;

                private:
                    //! cross section fit parameter 1, unitless
                    BoxValue m_boxCxin1;
                    //! cross section fit parameter 2, unitless
                    BoxValue m_boxCxin2;
                    //! cross section fit parameter 3, unitless
                    BoxValue m_boxCxin3;
                    //! cross section fit parameter 4, unitless
                    BoxValue m_boxCxin4;
                    //! cross section fit parameter 5, unitless
                    BoxValue m_boxCxin5;
                    //! cross section fit parameter 6, unitless
                    BoxValue m_boxCxin6;
                    //! cross section fit parameter 7, unitless
                    BoxValue m_boxCxin7;
                    //! cross section fit parameter 8, unitless
                    BoxValue m_boxCxin8;

                public:
                    /** constructor
                     *
                     * @attention transition data must be sorted block-wise ascending by lower/upper
                     *  atomic state and secondary ascending by upper/lower atomic state.
                     *
                     * @param boxCxin1 cross section fit parameter 1
                     * @param boxCxin2 cross section fit parameter 2
                     * @param boxCxin3 cross section fit parameter 3
                     * @param boxCxin4 cross section fit parameter 4
                     * @param boxCxin5 cross section fit parameter 5
                     * @param boxCxin4 cross section fit parameter 6
                     * @param boxCxin5 cross section fit parameter 7
                     * @param boxCxin5 cross section fit parameter 8
                     * @param boxLowerConfigNumber configNumber of the lower(lower excitation energy) state of the
                     * transition
                     * @param boxUpperConfigNumber configNumber of the upper(higher excitation energy) state of the
                     * transition
                     * @param T_numberTransitions number of atomic bound-free transitions stored
                     */
                    BoundFreeTransitionDataBox(
                        BoxValue boxCxin1,
                        BoxValue boxCxin2,
                        BoxValue boxCxin3,
                        BoxValue boxCxin4,
                        BoxValue boxCxin5,
                        BoxValue boxCxin6,
                        BoxValue boxCxin7,
                        BoxValue boxCxin8,
                        BoxConfigNumber boxLowerConfigNumber,
                        BoxConfigNumber boxUpperConfigNumber,
                        uint32_t numberTransitions)
                        : m_boxCxin1(boxCxin1)
                        , m_boxCxin2(boxCxin2)
                        , m_boxCxin3(boxCxin3)
                        , m_boxCxin4(boxCxin4)
                        , m_boxCxin5(boxCxin5)
                        , m_boxCxin6(boxCxin6)
                        , m_boxCxin7(boxCxin7)
                        , m_boxCxin8(boxCxin8)
                        , TransitionDataBox(boxLowerConfigNumber, boxUpperConfigNumber, numberTransitions)
                    {
                    }

                    /** store transition in data box
                     *
                     * @attention do not forget to call syncToDevice() on the
                     *  corresponding buffer, or the state is only stored on the host side.
                     * @attention needs to fulfill all ordering and content assumptions of constructor!
                     *
                     * @param collectionIndex index of data box entry to rewrite
                     * @param tuple tuple containing data of transition
                     */
                    HINLINE void store(uint32_t const collectionIndex, S_BoundFreeTransitionTuple& tuple)
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex >= m_numberTransitions)
                        {
                            throw std::runtime_error("atomicPhysics ERROR: outside range call");
                            return;
                        }
                        m_boxCxin1[collectionIndex] = std::get<0>(tuple);
                        m_boxCxin2[collectionIndex] = std::get<1>(tuple);
                        m_boxCxin3[collectionIndex] = std::get<2>(tuple);
                        m_boxCxin4[collectionIndex] = std::get<3>(tuple);
                        m_boxCxin5[collectionIndex] = std::get<4>(tuple);
                        m_boxCxin6[collectionIndex] = std::get<5>(tuple);
                        m_boxCxin7[collectionIndex] = std::get<6>(tuple);
                        m_boxCxin8[collectionIndex] = std::get<7>(tuple);
                        storeTransitions(collectionIndex, std::get<8>(tuple), std::get<9>(tuple));
                    }

                    /// @todo find way to replace Cxin getters with single template function, Brian Marre, 2022

                    /** returns cross section fit parameter 1 of the transition
                     *
                     * @param collectionIndex ... collection index of transition
                     *
                     * @attention no range checks outside debug compile
                     */
                    HDINLINE TypeValue cxin1(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex >= m_numberTransition)
                        {
                            printf("atomicPhysics ERROR: outside range call cxin1\n");
                            return static_cast<ValueType>(0._X);
                        }
                        return m_boxCxin1(indexTransition);
                    }

                    /** returns cross section fit parameter 2 of the transition
                     *
                     * @param collectionIndex ... collection index of transition
                     *
                     * @attention no range checks outside debug compile
                     */
                    HDINLINE TypeValue cxin2(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex >= m_numberTransition)
                        {
                            printf("atomicPhysics ERROR: outside range call cxin2\n");
                            return static_cast<ValueType>(0._X);
                        }
                        return m_boxCxin2(indexTransition);
                    }

                    /** returns cross section fit parameter 3 of the transition
                     *
                     * @param collectionIndex ... collection index of transition
                     *
                     * @attention no range checks outside debug compile
                     */
                    HDINLINE TypeValue cxin3(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex >= m_numberTransition)
                        {
                            printf("atomicPhysics ERROR: outside range call cxin3\n");
                            return static_cast<ValueType>(0._X);
                        }
                        return m_boxCxin3(indexTransition);
                    }

                    /** returns cross section fit parameter 4 of the transition
                     *
                     * @param collectionIndex ... collection index of transition
                     *
                     * @attention no range checks outside debug compile
                     */
                    HDINLINE TypeValue cxin4(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex >= m_numberTransition)
                        {
                            printf("atomicPhysics ERROR: outside range call cxin4\n");
                            return static_cast<ValueType>(0._X);
                        }
                        return m_boxCxin4(indexTransition);
                    }

                    /** returns cross section fit parameter 5 of the transition
                     *
                     * @param collectionIndex ... collection index of transition
                     *
                     * @attention no range checks outside debug compile
                     */
                    HDINLINE TypeValue cxin5(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex >= m_numberTransition)
                        {
                            printf("atomicPhysics ERROR: outside range call cxin5\n");
                            return static_cast<ValueType>(0._X);
                        }
                        return m_boxCxin5(indexTransition);
                    }

                   /** returns cross section fit parameter 6 of the transition
                    *
                    * @param collectionIndex ... collection index of transition
                    *
                    * @attention no range checks
                    */
                    HDINLINE TypeValue cxin6(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex >= m_numberTransition)
                        {
                            printf("atomicPhysics ERROR: outside range call cxin6\n");
                            return static_cast<ValueType>(0._X);
                        }
                        return m_boxCxin6(indexTransition);
                    }

                    /** returns cross section fit parameter 7 of the transition
                     *
                     * @param collectionIndex ... collection index of transition
                     *
                     * @attention no range checks outside debug compile
                     */
                    HDINLINE TypeValue cxin7(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex >= m_numberTransition)
                        {
                            printf("atomicPhysics ERROR: outside range call cxin7\n");
                            return static_cast<ValueType>(0._X);
                        }
                        return m_boxCxin7(indexTransition);
                    }

                    /** returns cross section fit parameter 8 of the transition
                     *
                     * @param collectionIndex ... collection index of transition
                     *
                     * @attention no range checks outside debug compile
                     */
                    HDINLINE TypeValue cxin8(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex >= m_numberTransition)
                        {
                            printf("atomicPhysics ERROR: outside range call cxin8\n");
                            return static_cast<ValueType>(0._X);
                        }
                        return m_boxCxin8(indexTransition);
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
                class BoundFreeTransitionDataBuffer : public TransitionDataBuffer< T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber>
                {
                    std::unique_ptr<BufferValue> bufferCxin1;
                    std::unique_ptr<BufferValue> bufferCxin2;
                    std::unique_ptr<BufferValue> bufferCxin3;
                    std::unique_ptr<BufferValue> bufferCxin4;
                    std::unique_ptr<BufferValue> bufferCxin5;
                    std::unique_ptr<BufferValue> bufferCxin6;
                    std::unique_ptr<BufferValue> bufferCxin7;
                    std::unique_ptr<BufferValue> bufferCxin8;

                public:
                    /** buffer corresponding to the above dataBox object
                     *
                     * @param numberAtomicStates number of atomic states, and number of buffer entries
                     */
                    HINLINE BoundFreeTransitionDataBuffer(uint32_t numberBoundFreeTransitions)
                        : TransitionDataBuffer(numberBoundFreeTransitions)
                    {
                        auto const guardSize = pmacc::DataSpace<1>::create(0);
                        auto const layoutBoundFreeTransitions = pmacc::GridLayout<1>(numberBoundFreeTransitions, guardSize);

                        bufferCxin1.reset(new BufferValue(layoutBoundFreeTransitions));
                        bufferCxin2.reset(new BufferValue(layoutBoundFreeTransitions));
                        bufferCxin3.reset(new BufferValue(layoutBoundFreeTransitions));
                        bufferCxin4.reset(new BufferValue(layoutBoundFreeTransitions));
                        bufferCxin5.reset(new BufferValue(layoutBoundFreeTransitions));
                        bufferCxin6.reset(new BufferValue(layoutBoundFreeTransitions));
                        bufferCxin7.reset(new BufferValue(layoutBoundFreeTransitions));
                        bufferCxin8.reset(new BufferValue(layoutBoundFreeTransitions));
                    }

                    HINLINE BoundFreeTransitionDataBox< T_DataBoxType, T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber> getHostDataBox()
                    {
                        return BoundFreeTransitionDataBox<
                            T_DataBoxType,
                            T_Number,
                            T_Value,
                            T_ConfigNumberDataType,
                            T_atomicNumber>(
                            bufferCxin1->getHostBuffer().getDataBox(),
                            bufferCxin2->getHostBuffer().getDataBox(),
                            bufferCxin3->getHostBuffer().getDataBox(),
                            bufferCxin4->getHostBuffer().getDataBox(),
                            bufferCxin5->getHostBuffer().getDataBox(),
                            bufferCxin6->getHostBuffer().getDataBox(),
                            bufferCxin7->getHostBuffer().getDataBox(),
                            bufferCxin8->getHostBuffer().getDataBox(),
                            bufferLowerConfigNumber->getHostBuffer().getDataBox(),
                            bufferUpperConfigNumber->getHostBuffer().getDataBox(),
                            m_numberTransitions);
                    }

                    HINLINE BoundFreeTransitionDataBox< T_DataBoxType, T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber> getDeviceDataBox()
                    {
                        return BoundFreeTransitionDataBox<
                            T_DataBoxType,
                            T_Number,
                            T_Value,
                            T_ConfigNumberDataType,
                            T_atomicNumber>(
                            bufferCxin1->getDeviceBuffer().getDataBox(),
                            bufferCxin2->getDeviceBuffer().getDataBox(),
                            bufferCxin3->getDeviceBuffer().getDataBox(),
                            bufferCxin4->getDeviceBuffer().getDataBox(),
                            bufferCxin5->getDeviceBuffer().getDataBox(),
                            bufferCxin6->getDeviceBuffer().getDataBox(),
                            bufferCxin7->getDeviceBuffer().getDataBox(),
                            bufferCxin8->getDeviceBuffer().getDataBox(),
                            bufferLowerConfigNumber->getDeviceBuffer().getDataBox(),
                            bufferUpperConfigNumber->getDeviceBuffer().getDataBox(),
                            m_numberTransitions);
                    }

                    HINLINE void syncToDevice()
                    {
                        bufferCxin1->hostToDevice();
                        bufferCxin2->hostToDevice();
                        bufferCxin3->hostToDevice();
                        bufferCxin4->hostToDevice();
                        bufferCxin5->hostToDevice();
                        bufferCxin6->hostToDevice();
                        bufferCxin7->hostToDevice();
                        bufferCxin8->hostToDevice();
                        syncToDevice_BaseClass();
                    }

                };

            } // namespace atomicData
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
