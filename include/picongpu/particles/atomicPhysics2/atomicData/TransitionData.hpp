/* Copyright 2020-2022 Brian Marre
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

#include "picongpu/particles/atomicPhysics2/atomicData/DataBox.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/DataBuffer.hpp"

#include <cstdint>
#include <memory>

/** @file implements base class of transitions property data
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
                template<
                    typename T_DataBoxType,
                    typename T_Number,
                    typename T_Value,
                    typename T_ConfigNumberDataType,
                    uint8_t T_atomicNumber>
                class TransitionDataBox : public DataBox<T_DataBoxType, T_Number, T_Value, T_atomicNumber>
                {
                public:
                    using Idx = T_ConfigNumberDataType;
                    using BoxConfigNumber = T_DataBoxType<T_ConfigNumberDataType>;

                protected:
                    //! configNumber of the lower(lower excitation energy) state of the transition
                    BoxConfigNumber m_boxLowerConfigNumber;
                    //! configNumber of the upper(higher excitation energy) state of the transition
                    BoxConfigNumber m_boxUpperConfigNumber;

                    uint32_t numberTransitions;

                public:
                    /** constructor
                     *
                     * @attention transition data must be sorted block-wise by atomic state
                     *  and secondary ascending by upper configNumber.
                     * @param boxLowerConfigNumber configNumber of the lower(lower excitation energy) state of the transition
                     * @param boxUpperConfigNumber configNumber of the upper(higher excitation energy) state of the transition
                     */
                    TransitionDataBox(
                        BoxConfigNumber boxLowerConfigNumber,
                        BoxConfigNumber boxUpperConfigNumber,
                        uint32_t numberTransitions)
                        : m_boxLowerConfigNumber(boxLowerConfigNumber)
                        , m_boxUpperConfigNumber(boxUpperConfigNumber)
                        , m_numberTransitions(numberTransitions)
                    {
                    }

                     /** returns collection index of transition in databox
                     *
                     * @param lowerConfigNumber configNumber of lower state
                     * @param upperConfigNumber configNumber of upper state
                     * @param startIndexBlock start collection of search
                     * @param numberOfTransitionsInBlock number of transitions to search
                     *
                     * @attention this search is slow, performant access should use collectionIndices directly
                     *
                     * @return returns numberTransitions if transition not found
                     *
                     * @todo : replace linear search, Brian Marre, 2022
                     */
                    HDINLINE uint32_t findTransitionCollectionIndex(
                        Idx const lowerConfigNumber,
                        Idx const upperConfigNumber,
                        uint32_t const numberOfTransitionsInBlock,
                        uint32_t const startIndexBlock=0u) const
                    {
                        // search in corresponding block in transitions box
                        for(uint32_t i = 0u; i < numberOfTransitionsInBlock; i++)
                        {
                            if( (m_boxLowerConfigNumber(i + startIndexBlock) == lowerConfigNumber)
                                    and (m_boxUpperConfigNumber(i + startIndexBlock) == upperConfigNumber) )
                                return i + startIndexBlock;
                        }

                        return numberTransitions;
                    }

                   /** returns upper states configNumber of the transition
                    *
                    * @param collectionIndex ... collection index of transition
                    *
                    * @attention no range checks
                    */
                    HDINLINE Idx getUpperConfigNumberTransition(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex < numberTransitions)
                            return m_boxUpperConfigNumber(collectionIndex);
                        return 0u;
                    }

                   /** returns lower states configNumber of the transition
                    *
                    * @param collectionIndex ... collection index of transition
                    *
                    * @attention no range checks
                    */
                    HDINLINE Idx getLowerConfigNumberTransition(uint32_t const collectionIndex) const
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex < numberTransitions)
                            return m_boxLowerConfigNumber(collectionIndex);
                        return 0u;
                    }

                    HDINLINE uint32_t getNumberOfTransitionsTotal()
                    {
                        return numberTransitions;
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
                class TransitionDataBuffer : public DataBuffer<T_Number, T_Value, T_atomicNumber>
                {
                public:
                    using Idx = T_ConfigNumberDataType;
                    using BufferConfigNumber = pmacc::HostDeviceBuffer<T_ConfigNumberDataType, 1u>;

                protected:
                    std::unique_ptr< BufferConfigNumber > bufferLowerConfigNumber;
                    std::unique_ptr< BufferConfigNumber > bufferUpperConfigNumber;

                    uint32_t m_numberTransitions;

                public:
                    /** buffer corresponding to the above dataBox object
                     *
                     * @param numberAtomicStates number of atomic states, and number of buffer entries
                     */
                    HINLINE TransitionDataBuffer(uint32_t numberTransitions)
                    {
                        auto const guardSize = pmacc::DataSpace<1>::create(0);
                        auto const layoutTransitions = pmacc::GridLayout<1>(numberTransitions, guardSize);

                        bufferNumberPhysicalTransitionsTotal.reset( new BufferConfigNumber(layoutTransitions));
                        m_numberTransitions = numberTransitions;
                    }

                    HINLINE TransitionDataBox<T_DataBoxType, T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber> getHostDataBox()
                    {
                        return TransitionDataBox<T_DataBoxType, T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber>(
                            bufferLowerConfigNumber->getHostBuffer().getDataBox(),
                            bufferUpperConfigNumber->getHostBuffer().getDataBox(),
                            numberTransitions);
                    }

                    HINLINE TransitionDataBox<T_DataBoxType, T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber> getDeviceDataBox()
                    {
                        return TransitionDataBox<T_DataBoxType, T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber>(
                            bufferLowerConfigNumber->getDeviceBuffer().getDataBox(),
                            bufferUpperConfigNumber->getDeviceBuffer().getDataBox(),
                            numberTransitions);
                    }

                    HINLINE void syncToDevice()
                    {
                        bufferLowerConfigNumber->hostToDevice();
                        bufferUpperConfigNumber->hostToDevice();
                    }

                };

            } // namespace atomicData
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
