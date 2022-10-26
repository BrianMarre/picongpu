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
#include "picongpu/particles/atomicPhysics2/atomicData/DataBox.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/DataBuffer.hpp"

#include <cstdint>
#include <memory>

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
                    BoxValue m_boxCinx1;
                    //! cross section fit parameter 2, unitless
                    BoxValue m_boxCinx2;
                    //! cross section fit parameter 3, unitless
                    BoxValue m_boxCinx3;
                    //! cross section fit parameter 4, unitless
                    BoxValue m_boxCinx4;
                    //! cross section fit parameter 5, unitless
                    BoxValue m_boxCinx5;
                    //! cross section fit parameter 6, unitless
                    BoxValue m_boxCinx6;
                    //! cross section fit parameter 7, unitless
                    BoxValue m_boxCinx7;
                    //! cross section fit parameter 8, unitless
                    BoxValue m_boxCinx8;

                public:
                    /** constructor
                     *
                     * @attention transition data must be sorted block-wise by lower atomic state
                     *  and secondary ascending by upper configNumber.
                     *
                     * @param boxCinx1 cross section fit parameter 1
                     * @param boxCinx2 cross section fit parameter 2
                     * @param boxCinx3 cross section fit parameter 3
                     * @param boxCinx4 cross section fit parameter 4
                     * @param boxCinx5 cross section fit parameter 5
                     * @param boxCinx4 cross section fit parameter 6
                     * @param boxCinx5 cross section fit parameter 7
                     * @param boxCinx5 cross section fit parameter 8
                     * @param boxLowerConfigNumber configNumber of the lower(lower excitation energy) state of the transition
                     * @param boxUpperConfigNumber configNumber of the upper(higher excitation energy) state of the transition
                     * @param T_numberTransitions number of atomic bound-free transitions stored
                     */
                    BoundFreeTransitionDataBox(
                        BoxValue boxCinx1,
                        BoxValue boxCinx2,
                        BoxValue boxCinx3,
                        BoxValue boxCinx4,
                        BoxValue boxCinx5,
                        BoxValue boxCinx6,
                        BoxValue boxCinx7,
                        BoxValue boxCinx8,
                        BoxConfigNumber boxLowerConfigNumber,
                        BoxConfigNumber boxUpperConfigNumber,
                        uint32_t numberTransitions)
                        : m_boxCinx1(boxCinx1)
                        , m_boxCinx2(boxCinx2)
                        , m_boxCinx3(boxCinx3)
                        , m_boxCinx4(boxCinx4)
                        , m_boxCinx5(boxCinx5)
                        , m_boxCinx6(boxCinx6)
                        , m_boxCinx7(boxCinx7)
                        , m_boxCinx8(boxCinx8)
                        , TransitionDataBox(boxLowerConfigNumber, boxUpperConfigNumber, numberTransitions)
                        {
                        }

                        /** store transition in data box
                         *
                         * @attention do not forget to call syncToDevice() on the
                         *  corresponding buffer, or the state is only added on the host side.
                         * @attention needs to fulfill all ordering and content assumptions of constructor!
                         *
                         * @param collectionIndex index of data box entry to rewrite
                         * @param tuple tuple containing data of transition
                         */
                        HINLINE void store(uint32_t const collectionIndex, S_BoundFreeTransitionTuple& tuple)
                        {
                            m_boxCinx1[collectionIndex] = std::get<0>(tuple);
                            m_boxCinx2[collectionIndex] = std::get<1>(tuple);
                            m_boxCinx3[collectionIndex] = std::get<2>(tuple);
                            m_boxCinx4[collectionIndex] = std::get<3>(tuple);
                            m_boxCinx5[collectionIndex] = std::get<4>(tuple);
                            m_boxCinx6[collectionIndex] = std::get<5>(tuple);
                            m_boxCinx7[collectionIndex] = std::get<6>(tuple);
                            m_boxCinx8[collectionIndex] = std::get<7>(tuple);
                            storeTransitions(collectionIndex, std::get<8>(tuple), std::get<9>(tuple));
                        }

                        /// @todo find way to replace cinx getters with single template function, Brian Marre, 2022

                        /** returns cross section fit parameter 1 of the transition
                         *
                         * @param collectionIndex ... collection index of transition
                         *
                         * @attention no range checks
                         */
                        HDINLINE TypeValue getCinx1(uint32_t const collectionIndex) const
                        {
                            // debug only
                            /// @todo find correct compile guard, Brian Marre, 2022
                            if(collectionIndex < numberTransitions)
                                return m_boxCinx1(indexTransition);
                            return static_cast<ValueType>(0._X);
                    }

                   /** returns cross section fit parameter 2 of the transition
                    *
                    * @param collectionIndex ... collection index of transition
                    *
                    * @attention no range checks
                    */
                    HDINLINE TypeValue getCinx2(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex < numberTransitions)
                            return m_boxCinx2(indexTransition);
                        return static_cast<ValueType>(0._X);
                    }

                   /** returns cross section fit parameter 3 of the transition
                    *
                    * @param collectionIndex ... collection index of transition
                    *
                    * @attention no range checks
                    */
                    HDINLINE TypeValue getCinx3(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex < numberTransitions)
                            return m_boxCinx3(indexTransition);
                        return static_cast<ValueType>(0._X);
                    }

                   /** returns cross section fit parameter 4 of the transition
                    *
                    * @param collectionIndex ... collection index of transition
                    *
                    * @attention no range checks
                    */
                    HDINLINE TypeValue getCinx4(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex < numberTransitions)
                            return m_boxCinx4(indexTransition);
                        return static_cast<ValueType>(0._X);
                    }

                   /** returns cross section fit parameter 5 of the transition
                    *
                    * @param collectionIndex ... collection index of transition
                    *
                    * @attention no range checks
                    */
                    HDINLINE TypeValue getCinx5(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex < numberTransitions)
                            return m_boxCinx5(indexTransition);
                        return static_cast<ValueType>(0._X);
                    }

                   /** returns cross section fit parameter 6 of the transition
                    *
                    * @param collectionIndex ... collection index of transition
                    *
                    * @attention no range checks
                    */
                    HDINLINE TypeValue getCinx6(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex < numberTransitions)
                            return m_boxCinx6(indexTransition);
                        return static_cast<ValueType>(0._X);
                    }

                   /** returns cross section fit parameter 7 of the transition
                    *
                    * @param collectionIndex ... collection index of transition
                    *
                    * @attention no range checks
                    */
                    HDINLINE TypeValue getCinx7(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex < numberTransitions)
                            return m_boxCinx7(indexTransition);
                        return static_cast<ValueType>(0._X);
                    }

                   /** returns cross section fit parameter 8 of the transition
                    *
                    * @param collectionIndex ... collection index of transition
                    *
                    * @attention no range checks
                    */
                    HDINLINE TypeValue getCinx8(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex < numberTransitions)
                            return m_boxCinx8(indexTransition);
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
                    typename T_DataBoxType,
                    typename T_Number,
                    typename T_Value,
                    typename T_ConfigNumberDataType,
                    uint8_t T_atomicNumber>
                class BoundFreeTransitionDataBuffer : public TransitionDataBuffer< T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber>
                {
                    std::unique_ptr< BufferValue > bufferCinx1;
                    std::unique_ptr< BufferValue > bufferCinx2;
                    std::unique_ptr< BufferValue > bufferCinx3;
                    std::unique_ptr< BufferValue > bufferCinx4;
                    std::unique_ptr< BufferValue > bufferCinx5;
                    std::unique_ptr< BufferValue > bufferCinx6;
                    std::unique_ptr< BufferValue > bufferCinx7;
                    std::unique_ptr< BufferValue > bufferCinx8;

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

                        bufferCinx1.reset( new BufferValue(layoutBoundFreeTransitions));
                        bufferCinx2.reset( new BufferValue(layoutBoundFreeTransitions));
                        bufferCinx3.reset( new BufferValue(layoutBoundFreeTransitions));
                        bufferCinx4.reset( new BufferValue(layoutBoundFreeTransitions));
                        bufferCinx5.reset( new BufferValue(layoutBoundFreeTransitions));
                        bufferCinx6.reset( new BufferValue(layoutBoundFreeTransitions));
                        bufferCinx7.reset( new BufferValue(layoutBoundFreeTransitions));
                        bufferCinx8.reset( new BufferValue(layoutBoundFreeTransitions));
                    }

                    HINLINE BoundFreeTransitionDataBox< T_DataBoxType, T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber> getHostDataBox()
                    {
                        return BoundFreeTransitionDataBox< T_DataBoxType, T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber>(
                            bufferCinx1->getHostBuffer().getDataBox(),
                            bufferCinx2->getHostBuffer().getDataBox(),
                            bufferCinx3->getHostBuffer().getDataBox(),
                            bufferCinx4->getHostBuffer().getDataBox(),
                            bufferCinx5->getHostBuffer().getDataBox(),
                            bufferCinx6->getHostBuffer().getDataBox(),
                            bufferCinx7->getHostBuffer().getDataBox(),
                            bufferCinx8->getHostBuffer().getDataBox(),
                            bufferLowerConfigNumber->getHostBuffer().getDataBox(),
                            bufferUpperConfigNumber->getHostBuffer().getDataBox(),
                            m_numberTransitions);
                    }

                    HINLINE BoundFreeTransitionDataBox< T_DataBoxType, T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber> getDeviceDataBox()
                    {
                        return BoundFreeTransitionDataBox< T_DataBoxType, T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber>(
                            bufferCinx1->getDeviceBuffer().getDataBox(),
                            bufferCinx2->getDeviceBuffer().getDataBox(),
                            bufferCinx3->getDeviceBuffer().getDataBox(),
                            bufferCinx4->getDeviceBuffer().getDataBox(),
                            bufferCinx5->getDeviceBuffer().getDataBox(),
                            bufferCinx6->getDeviceBuffer().getDataBox(),
                            bufferCinx7->getDeviceBuffer().getDataBox(),
                            bufferCinx8->getDeviceBuffer().getDataBox(),
                            bufferLowerConfigNumber->getDeviceBuffer().getDataBox(),
                            bufferUpperConfigNumber->getDeviceBuffer().getDataBox(),
                            m_numberTransitions);
                    }

                    HINLINE void syncToDevice()
                    {
                        bufferCinx1->hostToDevice();
                        bufferCinx2->hostToDevice();
                        bufferCinx3->hostToDevice();
                        bufferCinx4->hostToDevice();
                        bufferCinx5->hostToDevice();
                        bufferCinx6->hostToDevice();
                        bufferCinx7->hostToDevice();
                        bufferCinx8->hostToDevice();
                        syncToDevice_BaseClass();
                    }

                };

            } // namespace atomicData
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
