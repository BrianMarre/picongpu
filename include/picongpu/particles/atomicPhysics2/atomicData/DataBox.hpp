/* Copyright 2022 Brian Marre
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


namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics2
        {
            namespace atomicData
            {
                /** common interfaces of all data storage classes
                 *
                 * @tparam T_DataBoxType dataBox type used for storage
                 * @tparam T_Number dataType used for number storage, typically uint32_t
                 * @tparam T_Value dataType used for value storage, typically float_X
                 * @tparam T_ConfigNumberDataType dataType used for configNumber storage,
                 *      typically uint64_t
                 * @tparam T_atomicNumber atomic number of element this data corresponds to, eg. Cu -> 29
                 */
                template<
                    typename T_Number,
                    typename T_Value,
                    uint8_t T_atomicNumber // element
                    >
                class DataBox
                {
                public:
                    template<typename T_DataType>
                    using T_DataBoxType = pmacc::DataBox<pmacc::PitchedBox<T_DataType, 1u>>;

                    using BoxNumber = T_DataBoxType<T_Number>;
                    using BoxValue = T_DataBoxType<T_Value>;

                    using TypeNumber = T_Number;
                    using TypeValue = T_Value;

                    template<typename T_DataType>
                    using DataBoxType = T_DataBoxType<T_DataType>;

                    constexpr static uint8_t atomicNumber = T_atomicNumber;
                };

            } // namespace atomicData
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
