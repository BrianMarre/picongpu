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

#include "picongpu/particles/atomicPhysics2/stateRepresentation/ConfigNumber.hpp"

#include <pmacc/math/Vector.hpp>

#include <cstdint>
#include <list>
#include <utility>
#include <array>
#include <string>
#include <iostream>

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics2
        {
            namespace stateRepresentation
            {
                namespace debug
                {

                    using DataType = uint64_t;
                    constexpr uint8_t numberLevels = 10u;
                    constexpr uint8_t atomicNumber = 18u;

                    using Config = ConfigNumber<DataType, numberLevels, atomicNumber>;
                    using LevelVector = std::array<uint8_t, numberLevels>;
                    using pmaccVector = pmacc::math::Vector<uint8_t, numberLevels>;

                    using TestExample = std::pair<DataType, LevelVector>;

                    template<uint8_t T_numberLevels>
                    std::string levelVectorToString(LevelVector levelVector)
                    {
                        std::string result = "( " + std::to_string(levelVector[0]);

                        for (uint8_t i = 1u; i < T_numberLevels; i++)
                        {
                            result += ", " + std::to_string(levelVector[i]);
                        }
                        result += " )";

                        return result;
                    }

                    template<uint8_t T_numberLevels>
                    void convertToLevelVector( LevelVector& levelVector, pmaccVector vector)
                    {
                        uint8_t temp;

                        for (uint8_t i = 0u; i < T_numberLevels; i++)
                        {
                            temp = static_cast<uint8_t>(vector[i]);
                            levelVector[i] = temp;
                        }
                    }

                    //! debug only, configNumber conversion methods tests, cpu only
                    bool testConfigNumberConversionMethods()
                    {
                        // config options
                        using DataType = uint64_t;
                        constexpr uint8_t numberLevels = 10u;
                        constexpr uint8_t atomicNumber = 18u;

                        using Config = ConfigNumber<DataType, numberLevels, atomicNumber>;
                        using LevelVector = std::array<uint8_t, numberLevels>;
                        using pmaccVector = pmacc::math::Vector<uint8_t, numberLevels>;

                        using TestExample = std::pair<DataType, LevelVector>;

                        // testCases
                        std::list<TestExample> testExamples = {
                            // configNumber            levelVector
                            // standard examples
                            TestExample{9779u,         LevelVector{2, 1, 1, 0,  1,  0,  0, 0, 0, 0}},
                            TestExample{66854705u,     LevelVector{2, 1, 1, 0,  0,  0,  0, 1, 0, 0}},
                            // high value test examples
                            TestExample{24134536956u,  LevelVector{0, 1, 0, 0,  0,  0,  0, 0, 0, 1}},
                            TestExample{24134537168u,  LevelVector{2, 8, 7, 0,  0,  0,  0, 0, 0, 1}},
                            TestExample{24134537139u,  LevelVector{0, 8, 6, 0,  0,  0,  0, 0, 0, 1}},
                            TestExample{434421665154u, LevelVector{0, 0, 0, 0,  0,  0,  0, 0, 0, 18}},
                            // all levels test example
                            TestExample{25475344564u,  LevelVector{1, 1, 1, 1,  1,  1,  1, 1, 1, 1}}};

                        bool pass = true;

                        DataType knownConfigNumber;
                        LevelVector knownLevelVector;
                        DataType configNumber;
                        pmaccVector levelVectorTemp;
                        LevelVector levelVector;

                        for (TestExample const& testCase : testExamples)
                        {
                            // good Results
                            knownConfigNumber = std::get<0>(testCase);
                            knownLevelVector = std::get<1>(testCase);

                            std::cout << "Test: (" << knownConfigNumber << ", ";
                            std::cout << levelVectorToString<numberLevels>(knownLevelVector) << ")" << std::endl;

                            // getLevelVector()
                            levelVectorTemp = Config::getLevelVector(knownConfigNumber);
                            convertToLevelVector<numberLevels>(levelVector, levelVectorTemp);
                            pass = pass and (levelVector == knownLevelVector);
                            std::cout << "\t getLevelVector(): ";
                            std::cout << levelVectorToString<numberLevels>(levelVector);
                            std::cout << std::endl;

                            // getIonizationState()
                            uint8_t numberElectrons = 0u;
                            for (uint8_t i = 0u; i< numberLevels; i++)
                            {
                                numberElectrons += levelVector[i];
                            }
                            pass = pass and (
                                (atomicNumber - numberElectrons) == Config::getIonizationState(knownConfigNumber)
                                );

                            std::cout << "\t getIonizationState(): " << std::to_string(atomicNumber - numberElectrons);
                            std::cout << " =?= (returnValue:) ";
                            std::cout << std::to_string(
                                static_cast<uint16_t>(Config::getIonizationState(knownConfigNumber)));
                            std::cout << std::endl;

                            // getAtomicConfigNumber()
                            for (uint8_t i = 0u; i< numberLevels; i++)
                            {
                                levelVectorTemp[i] = static_cast<uint8_t>(knownLevelVector[i]); // reuse
                            }
                            pass = pass and ( knownConfigNumber == Config::getAtomicConfigNumber(levelVectorTemp));

                            std::cout << "\t getAtomicConfigNumber(): " << knownConfigNumber;
                            std::cout << " =?= " << Config::getAtomicConfigNumber(levelVectorTemp) << std::endl;
                        }

                        if (pass)
                            std::cout << "Success"<< std::endl;
                        else
                            std::cout << "Fail"<< std::endl;

                        return pass;
                    }

                } // namespace debug
            } // namespace stateRepresentation
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu