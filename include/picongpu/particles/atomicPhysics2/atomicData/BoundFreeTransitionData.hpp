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
                    typename T_Number,
                    typename T_Value,
                    typename T_ConfigNumberDataType,
                    uint8_t T_atomicNumber>
                class BoundFreeTransitionDataBox :
                    public TransitionDataBox<
                        T_Number,
                        T_Value,
                        T_ConfigNumberDataType,
                        T_atomicNumber>
                {
                public:
                    using S_TransitionDataBox = TransitionDataBox<
                        T_Number,
                        T_Value,
                        T_ConfigNumberDataType,
                        T_atomicNumber>;
                    using S_BoundFreeTransitionTuple = BoundFreeTransitionTuple<typename S_TransitionDataBox::TypeValue, typename S_TransitionDataBox::Idx>;

                private:
                    //! cross section fit parameter 1, unitless
                    typename S_TransitionDataBox::BoxValue m_boxCxin1;
                    //! cross section fit parameter 2, unitless
                    typename S_TransitionDataBox::BoxValue m_boxCxin2;
                    //! cross section fit parameter 3, unitless
                    typename S_TransitionDataBox::BoxValue m_boxCxin3;
                    //! cross section fit parameter 4, unitless
                    typename S_TransitionDataBox::BoxValue m_boxCxin4;
                    //! cross section fit parameter 5, unitless
                    typename S_TransitionDataBox::BoxValue m_boxCxin5;
                    //! cross section fit parameter 6, unitless
                    typename S_TransitionDataBox::BoxValue m_boxCxin6;
                    //! cross section fit parameter 7, unitless
                    typename S_TransitionDataBox::BoxValue m_boxCxin7;
                    //! cross section fit parameter 8, unitless
                    typename S_TransitionDataBox::BoxValue m_boxCxin8;

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
                        typename S_TransitionDataBox::BoxValue boxCxin1,
                        typename S_TransitionDataBox::BoxValue boxCxin2,
                        typename S_TransitionDataBox::BoxValue boxCxin3,
                        typename S_TransitionDataBox::BoxValue boxCxin4,
                        typename S_TransitionDataBox::BoxValue boxCxin5,
                        typename S_TransitionDataBox::BoxValue boxCxin6,
                        typename S_TransitionDataBox::BoxValue boxCxin7,
                        typename S_TransitionDataBox::BoxValue boxCxin8,
                        typename S_TransitionDataBox::BoxConfigNumber boxLowerConfigNumber,
                        typename S_TransitionDataBox::BoxConfigNumber boxUpperConfigNumber,
                        uint32_t numberTransitions)
                        : m_boxCxin1(boxCxin1)
                        , m_boxCxin2(boxCxin2)
                        , m_boxCxin3(boxCxin3)
                        , m_boxCxin4(boxCxin4)
                        , m_boxCxin5(boxCxin5)
                        , m_boxCxin6(boxCxin6)
                        , m_boxCxin7(boxCxin7)
                        , m_boxCxin8(boxCxin8)
                        , TransitionDataBox<
                            T_Number,
                            T_Value,
                            T_ConfigNumberDataType,
                            T_atomicNumber>(boxLowerConfigNumber, boxUpperConfigNumber, numberTransitions)
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
                        if(collectionIndex >= this->m_numberTransitions)
                        {
                            throw std::runtime_error("atomicPhysics ERROR: outside range call store bound-bound");
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
                        this->storeTransition(collectionIndex, std::get<8>(tuple), std::get<9>(tuple));
                    }

                    /// @todo find way to replace Cxin getters with single template function, Brian Marre, 2022

                    /** returns cross section fit parameter 1 of the transition
                     *
                     * @param collectionIndex ... collection index of transition
                     *
                     * @attention no range checks outside debug compile
                     */
                    HDINLINE typename S_TransitionDataBox::TypeValue cxin1(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex >= this->m_numberTransitions)
                        {
                            printf("atomicPhysics ERROR: outside range call cxin1\n");
                            return static_cast<typename S_TransitionDataBox::TypeValue>(0._X);
                        }
                        return m_boxCxin1(collectionIndex);
                    }

                    /** returns cross section fit parameter 2 of the transition
                     *
                     * @param collectionIndex ... collection index of transition
                     *
                     * @attention no range checks outside debug compile
                     */
                    HDINLINE typename S_TransitionDataBox::TypeValue cxin2(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex >= this->m_numberTransitions)
                        {
                            printf("atomicPhysics ERROR: outside range call cxin2\n");
                            return static_cast<typename S_TransitionDataBox::TypeValue>(0._X);
                        }
                        return m_boxCxin2(collectionIndex);
                    }

                    /** returns cross section fit parameter 3 of the transition
                     *
                     * @param collectionIndex ... collection index of transition
                     *
                     * @attention no range checks outside debug compile
                     */
                    HDINLINE typename S_TransitionDataBox::TypeValue cxin3(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex >= this->m_numberTransitions)
                        {
                            printf("atomicPhysics ERROR: outside range call cxin3\n");
                            return static_cast<typename S_TransitionDataBox::TypeValue>(0._X);
                        }
                        return m_boxCxin3(collectionIndex);
                    }

                    /** returns cross section fit parameter 4 of the transition
                     *
                     * @param collectionIndex ... collection index of transition
                     *
                     * @attention no range checks outside debug compile
                     */
                    HDINLINE typename S_TransitionDataBox::TypeValue cxin4(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex >= this->m_numberTransitions)
                        {
                            printf("atomicPhysics ERROR: outside range call cxin4\n");
                            return static_cast<typename S_TransitionDataBox::TypeValue>(0._X);
                        }
                        return m_boxCxin4(collectionIndex);
                    }

                    /** returns cross section fit parameter 5 of the transition
                     *
                     * @param collectionIndex ... collection index of transition
                     *
                     * @attention no range checks outside debug compile
                     */
                    HDINLINE typename S_TransitionDataBox::TypeValue cxin5(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex >= this->m_numberTransitions)
                        {
                            printf("atomicPhysics ERROR: outside range call cxin5\n");
                            return static_cast<typename S_TransitionDataBox::TypeValue>(0._X);
                        }
                        return m_boxCxin5(collectionIndex);
                    }

                   /** returns cross section fit parameter 6 of the transition
                    *
                    * @param collectionIndex ... collection index of transition
                    *
                    * @attention no range checks
                    */
                    HDINLINE typename S_TransitionDataBox::TypeValue cxin6(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex >= this->m_numberTransitions)
                        {
                            printf("atomicPhysics ERROR: outside range call cxin6\n");
                            return static_cast<typename S_TransitionDataBox::TypeValue>(0._X);
                        }
                        return m_boxCxin6(collectionIndex);
                    }

                    /** returns cross section fit parameter 7 of the transition
                     *
                     * @param collectionIndex ... collection index of transition
                     *
                     * @attention no range checks outside debug compile
                     */
                    HDINLINE typename S_TransitionDataBox::TypeValue cxin7(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex >= this->m_numberTransitions)
                        {
                            printf("atomicPhysics ERROR: outside range call cxin7\n");
                            return static_cast<typename S_TransitionDataBox::TypeValue>(0._X);
                        }
                        return m_boxCxin7(collectionIndex);
                    }

                    /** returns cross section fit parameter 8 of the transition
                     *
                     * @param collectionIndex ... collection index of transition
                     *
                     * @attention no range checks outside debug compile
                     */
                    HDINLINE typename S_TransitionDataBox::TypeValue cxin8(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex >= this->m_numberTransitions)
                        {
                            printf("atomicPhysics ERROR: outside range call cxin8\n");
                            return static_cast<typename S_TransitionDataBox::TypeValue>(0._X);
                        }
                        return m_boxCxin8(collectionIndex);
                    }

                };

                /** complementing buffer class
                 *
                 * @tparam T_Number dataType used for number storage, typically uint32_t
                 * @tparam T_Value dataType used for value storage, typically float_X
                 * @tparam T_atomicNumber atomic number of element this data corresponds to, eg. Cu -> 29
                 */
                template<
                    typename T_Number,
                    typename T_Value,
                    typename T_ConfigNumberDataType,
                    uint8_t T_atomicNumber>
                class BoundFreeTransitionDataBuffer : public TransitionDataBuffer< T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber>
                {
                public:
                    using S_TransitionDataBuffer = TransitionDataBuffer< T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber>;
                private:
                    std::unique_ptr<typename S_TransitionDataBuffer::BufferValue> bufferCxin1;
                    std::unique_ptr<typename S_TransitionDataBuffer::BufferValue> bufferCxin2;
                    std::unique_ptr<typename S_TransitionDataBuffer::BufferValue> bufferCxin3;
                    std::unique_ptr<typename S_TransitionDataBuffer::BufferValue> bufferCxin4;
                    std::unique_ptr<typename S_TransitionDataBuffer::BufferValue> bufferCxin5;
                    std::unique_ptr<typename S_TransitionDataBuffer::BufferValue> bufferCxin6;
                    std::unique_ptr<typename S_TransitionDataBuffer::BufferValue> bufferCxin7;
                    std::unique_ptr<typename S_TransitionDataBuffer::BufferValue> bufferCxin8;

                public:
                    /** buffer corresponding to the above dataBox object
                     *
                     * @param numberAtomicStates number of atomic states, and number of buffer entries
                     */
                    HINLINE BoundFreeTransitionDataBuffer(uint32_t numberBoundFreeTransitions)
                        : TransitionDataBuffer< T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber>(numberBoundFreeTransitions)
                    {
                        auto const guardSize = pmacc::DataSpace<1>::create(0);
                        auto const layoutBoundFreeTransitions = pmacc::GridLayout<1>(numberBoundFreeTransitions, guardSize).getDataSpaceWithoutGuarding();

                        bufferCxin1.reset(new typename S_TransitionDataBuffer::BufferValue(layoutBoundFreeTransitions, false));
                        bufferCxin2.reset(new typename S_TransitionDataBuffer::BufferValue(layoutBoundFreeTransitions, false));
                        bufferCxin3.reset(new typename S_TransitionDataBuffer::BufferValue(layoutBoundFreeTransitions, false));
                        bufferCxin4.reset(new typename S_TransitionDataBuffer::BufferValue(layoutBoundFreeTransitions, false));
                        bufferCxin5.reset(new typename S_TransitionDataBuffer::BufferValue(layoutBoundFreeTransitions, false));
                        bufferCxin6.reset(new typename S_TransitionDataBuffer::BufferValue(layoutBoundFreeTransitions, false));
                        bufferCxin7.reset(new typename S_TransitionDataBuffer::BufferValue(layoutBoundFreeTransitions, false));
                        bufferCxin8.reset(new typename S_TransitionDataBuffer::BufferValue(layoutBoundFreeTransitions, false));
                    }

                    HINLINE BoundFreeTransitionDataBox< T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber> getHostDataBox()
                    {
                        return BoundFreeTransitionDataBox<
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
                            this->bufferLowerConfigNumber->getHostBuffer().getDataBox(),
                            this->bufferUpperConfigNumber->getHostBuffer().getDataBox(),
                            this->m_numberTransitions);
                    }

                    HINLINE BoundFreeTransitionDataBox< T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber> getDeviceDataBox()
                    {
                        return BoundFreeTransitionDataBox<
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
                            this->bufferLowerConfigNumber->getDeviceBuffer().getDataBox(),
                            this->bufferUpperConfigNumber->getDeviceBuffer().getDataBox(),
                            this->m_numberTransitions);
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
                        this->syncToDevice_BaseClass();
                    }

                };

            } // namespace atomicData
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
